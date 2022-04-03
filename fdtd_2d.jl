#=
Copyright Alec Hammond 2022

Simple fdtd in 2d excercise.

Assumes TMᶻ polarization. Relevant physics:
    −σₐHx − μ∂Hx/∂t = ∂Ez/∂y
     σₐHy + μ∂Hy/∂t = ∂Ez/∂x
     σEz + ε∂Ez/∂t = ∂Hy/∂x - ∂Hx/∂y
where
    σₐ is the magnetic conductivity
    σ is the electric conductivity
    ε is the permittivity  
    μ is the permeability  

Define Ez in lower left corner: Ez(m,n)
Define Hy just to the right:    Hy(m+1/2,n)
Define Hx just above:           Hx(m,n+1/2)

Nodes at top of grid lack Hx
Nodes at right of grid lack Hy

References
-------------------
* https://eecs.wsu.edu/~schneidj/ufdtd/chap8.pdf

=#

const USE_GPU = false
using ImplicitGlobalGrid
using ParallelStencil
using ParallelStencil.FiniteDifferences2D
using Plots, Printf, Statistics

type = Float64
dims = 2
@static if USE_GPU
    @init_parallel_stencil(CUDA, type, dims);
else
    @init_parallel_stencil(Threads, type, dims);
end

@parallel function step_H!(Hx::Data.Array,Hy::Data.Array,Ez::Data.Array,dt::Data.Number,dx::Data.Number,dy::Data.Number)
    # assume σₐ=0
    # assume μ=1
    @all(Hx) = @all(Hx) - dt/dy*@d_ya(Ez)
    @all(Hy) = @all(Hy) + dt/dx*@d_xa(Ez)
    return
end

@parallel function step_E!(Hx::Data.Array,Hy::Data.Array,Ez::Data.Array,ε::Data.Array,dt::Data.Number, dx::Data.Number, dy::Data.Number)
    # assume σ=0
    @inn(Ez) = @inn(Ez) + (dt/dx./@inn(ε)).*(@d_xi(Hy)) - (dt/dy./@inn(ε)).*(@d_yi(Hx))
    return
end

@parallel_indices (ix,iy) function step_E_sources!(Ez::Data.Array,n::Int64,dt::Data.Number, dx::Data.Number, dy::Data.Number)
    # point source in center of cell
    if (ix==size(Ez,1)/2 && iy==size(Ez,2)/2)
        t = n*dt 
        Ez[ix,iy] = cos(20*t)*exp((t-5)^2)
    end
    return
end

function fdtd2D_step!(Hx::Data.Array,Hy::Data.Array,Ez::Data.Array,ε::Data.Array,n::Int64,dt::Data.Number, dx::Data.Number, dy::Data.Number)
    @parallel step_H!(Hx,Hy,Ez,dt,dx,dy)
    @parallel step_E!(Hx,Hy,Ez,ε,dt,dx,dy)
    @parallel step_E_sources!(Ez,n,dt,dx,dy)
    return
end

function fdtd2D()

    # Physics
    lx, ly = 1.0, 1.0;

    # Numerics
    C      = 0.5;        # Courant factor
    nx, ny = 256, 256;
    nt     = 250;

    # Derived numerics
    dx, dy    = lx/(nx-1), ly/(ny-1) # cell sizes
    dt        = min(dx,dy)*C

    # Array initializations
    Hx       = @zeros(nx, ny-1);
    Hy       = @zeros(nx-1, ny);
    Ez       = @zeros(nx, ny);
    ε        = @ones(nx, ny);
    @printf("Starting...\n")
    # Time loop
    for it = 1:nt
        if (it==11)  global wtime0 = Base.time()  end
        fdtd2D_step!(Hx,Hy,Ez,ε,it,dt,dx,dy)
    end
    @printf("Stopping...\n")

    # Performance
    wtime    = Base.time() - wtime0
    A_eff    = (2*2+2)/1e9*nx*ny*sizeof(Data.Number)  # Effective main memory access per iteration [GB] (Lower bound of required memory access: H and dHdτ have to be read and written (dHdτ for damping): 4 whole-array memaccess; B has to be read: 1 whole-array memaccess)
    wtime_it = wtime/(nt-10)                          # Execution time per iteration [s]
    T_eff    = A_eff/wtime_it                         # Effective memory throughput [GB/s]
    
    @printf("Total steps=%d, time=%1.3e sec (@ T_eff = %1.2f GB/s) \n", nt, wtime, round(T_eff, sigdigits=2))
    gr()
    hm = heatmap(Ez, color = :bluesreds, aspect_ratio=:equal, show = true)
    png("Test")
    return
end

fdtd2D()