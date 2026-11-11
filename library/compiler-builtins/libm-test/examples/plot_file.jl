"A quick script for plotting a list of floats.

Takes a path to a TOML file (Julia has builtin TOML support but not JSON) which
specifies a list of source files to plot. Plots are done with both a linear and
a log scale.

Requires [Makie] (specifically CairoMakie) for plotting.

[Makie]: https://docs.makie.org/stable/
"

using CairoMakie
using TOML

function main()::Nothing
    CairoMakie.activate!(px_per_unit = 10)
    config_path = ARGS[1]

    cfg = Dict()
    open(config_path, "r") do f
        cfg = TOML.parse(f)
    end

    out_dir = cfg["out_dir"]
    for input in cfg["input"]
        fn_name = input["function"]
        gen_name = input["generator"]
        input_file = input["input_file"]

        plot_one(input_file, out_dir, fn_name, gen_name)
    end
end

"Read inputs from a file, create both linear and log plots for one function"
function plot_one(
    input_file::String,
    out_dir::String,
    fn_name::String,
    gen_name::String,
)::Nothing
    fig = Figure()

    lin_out_file = joinpath(out_dir, "plot-$fn_name-$gen_name.png")
    log_out_file = joinpath(out_dir, "plot-$fn_name-$gen_name-log.png")

    # Map string function names to callable functions
    if fn_name == "cos"
        orig_func = cos
        xlims = (-6.0, 6.0)
        xlims_log = (-pi * 10, pi * 10)
    elseif fn_name == "cbrt"
        orig_func = cbrt
        xlims = (-2.0, 2.0)
        xlims_log = (-1000.0, 1000.0)
    elseif fn_name == "sqrt"
        orig_func = sqrt
        xlims = (-1.1, 6.0)
        xlims_log = (-1.1, 5000.0)
    else
        println("unrecognized function name `$fn_name`; update plot_file.jl")
        exit(1)
    end

    # Edge cases don't do much beyond +/-1, except for infinity.
    if gen_name == "edge_cases"
        xlims = (-1.1, 1.1)
        xlims_log = (-1.1, 1.1)
    end

    # Turn domain errors into NaN
    func(x) = map_or(x, orig_func, NaN)

    # Parse a series of X values produced by the generator
    inputs = readlines(input_file)
    gen_x = map((v) -> parse(Float32, v), inputs)

    do_plot(
        fig,
        gen_x,
        func,
        xlims[1],
        xlims[2],
        "$fn_name $gen_name (linear scale)",
        lin_out_file,
        false,
    )

    do_plot(
        fig,
        gen_x,
        func,
        xlims_log[1],
        xlims_log[2],
        "$fn_name $gen_name (log scale)",
        log_out_file,
        true,
    )
end

"Create a single plot"
function do_plot(
    fig::Figure,
    gen_x::Vector{F},
    func::Function,
    xmin::AbstractFloat,
    xmax::AbstractFloat,
    title::String,
    out_file::String,
    logscale::Bool,
)::Nothing where {F<:AbstractFloat}
    println("plotting $title")

    # `gen_x` is the values the generator produces. `actual_x` is for plotting a
    # continuous function.
    input_min = xmin - 1.0
    input_max = xmax + 1.0
    gen_x = filter((v) -> v >= input_min && v <= input_max, gen_x)
    markersize = length(gen_x) < 10_000 ? 6.0 : 4.0

    steps = 10_000
    if logscale
        r = LinRange(symlog10(input_min), symlog10(input_max), steps)
        actual_x = sympow10.(r)
        xscale = Makie.pseudolog10
    else
        actual_x = LinRange(input_min, input_max, steps)
        xscale = identity
    end

    gen_y = @. func(gen_x)
    actual_y = @. func(actual_x)

    ax = Axis(fig[1, 1], xscale = xscale, title = title)

    lines!(
        ax,
        actual_x,
        actual_y,
        color = (:lightblue, 0.6),
        linewidth = 6.0,
        label = "true function",
    )
    scatter!(
        ax,
        gen_x,
        gen_y,
        color = (:darkblue, 0.9),
        markersize = markersize,
        label = "checked inputs",
    )
    axislegend(ax, position = :rb, framevisible = false)

    save(out_file, fig)
    delete!(ax)
end

"Apply a function, returning the default if there is a domain error"
function map_or(input::AbstractFloat, f::Function, default::Any)::Union{AbstractFloat,Any}
    try
        return f(input)
    catch
        return default
    end
end

# Operations for logarithms that are symmetric about 0
C = 10
symlog10(x::Number) = sign(x) * (log10(1 + abs(x) / (10^C)))
sympow10(x::Number) = (10^C) * (10^x - 1)

main()
