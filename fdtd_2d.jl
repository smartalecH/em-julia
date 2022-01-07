#=
Simple fdtd in 2d excercise.

Assumes TMᶻ polarization. Relevant physics:
    −σₘHx − μ∂Hx/∂t = ∂Ez/∂y
     σₘHy + μ∂Hy/∂t = ∂Ez/∂x
     σ Ez + ε∂Ez/∂t = ∂Hy/∂x - ∂Hx/∂y

     Define Ez in lower left corner: Ez(m,n)
Define Hy just to the right:    Hy(m+1/2,n)
Define Hx just above:           Hx(m,n+1/2)

Nodes at top of grid lack Hx
Nodes at right of grid lack Hy
=#

const USE_GPU = false
#using ImplicitGlobalGrid
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
    @all(Hx) = @all(Hx) - dt/dy*@d_ya(Ez)
    @all(Hy) = @all(Hy) - dt/dx*@d_xa(Ez)
    return
end

@parallel function step_E!(Hx::Data.Array,Hy::Data.Array,Ez::Data.Array,ε::Data.Array,dt::Data.Number, dx::Data.Number, dy::Data.Number)
    @inn(Ez) = @inn(Ez) + (dt/dx./@inn(ε)).*(@d_xi(Hy)) - (dt/dy./@inn(ε)).*(@d_yi(Hx))
    return
end

@parallel_indices (ix,iy) function step_E_sources!(Ez::Data.Array,n::Int64,dt::Data.Number, dx::Data.Number, dy::Data.Number)
    t = 
    if (ix==size(Ez,1)/2 && iy==size(Ez,2)/2) Ez[ix,iy] += 1e-3; end;
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
    lx, ly = 10.0, 10.0;

    # Numerics
    C      = 0.5;        # Courant factor
    nx, ny = 256, 256;
    nt     = 50;

    # Derived numerics
    dx, dy    = lx/(nx-1), ly/(ny-1) # cell sizes
    dt        = min(dx,dy)*C

    # Array initializations
    Hx       = @zeros(nx, ny-1);
    Hy       = @zeros(nx-1, ny);
    Ez       = @zeros(nx, ny);
    ε        = @ones(nx, ny);

    # Time loop
    for it = 1:nt
        if (it==11)  global wtime0 = Base.time()  end
        fdtd2D_step!(Hx,Hy,Ez,ε,it,dt,dx,dy)
    end

    # Performance
    wtime    = Base.time() - wtime0
    A_eff    = (2*2+2)/1e9*nx*ny*sizeof(Data.Number)  # Effective main memory access per iteration [GB] (Lower bound of required memory access: H and dHdτ have to be read and written (dHdτ for damping): 4 whole-array memaccess; B has to be read: 1 whole-array memaccess)
    wtime_it = wtime/(nt-10)                          # Execution time per iteration [s]
    T_eff    = A_eff/wtime_it                         # Effective memory throughput [GB/s]
    
    @printf("Total steps=%d, time=%1.3e sec (@ T_eff = %1.2f GB/s) \n", nt, wtime, round(T_eff, sigdigits=2))
    gr()
    heatmap(Ez, color = :greys,show = true)
    png("Test")
    return
end

fdtd2D()