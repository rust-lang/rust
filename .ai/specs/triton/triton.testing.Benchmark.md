Triton

Getting Started

-   Installation
-   Tutorials

Python API

-   triton
-   triton.language
-   triton.testing
    -   triton.testing.Benchmark
        -   `Benchmark`
            -   `Benchmark.__init__()`
    -   triton.testing.do_bench
    -   triton.testing.do_bench_cudagraph
    -   triton.testing.perf_report
    -   triton.testing.assert_close
-   Triton Semantics

Triton MLIR Dialects

-   Triton MLIR Dialects and Ops

Programming Guide

-   Introduction
-   Related Work
-   Debugging Triton

Triton

-   
-   triton.testing
-   triton.testing.Benchmark
-   View page source

------------------------------------------------------------------------

# triton.testing.Benchmark¶

*class *triton.testing.Benchmark(*self*, *x_names: List\[str\]*, *x_vals: List\[Any\]*, *line_arg: str*, *line_vals: List\[Any\]*, *line_names: List\[str\]*, *plot_name: str*, *args: Dict\[str, Any\]*, *xlabel: str = \'\'*, *ylabel: str = \'\'*, *x_log: bool = False*, *y_log: bool = False*, *styles=None*)¶

:   This class is used by the `perf_report` function to generate line plots with a concise API.

    \_\_init\_\_(*self*, *x_names: List\[str\]*, *x_vals: List\[Any\]*, *line_arg: str*, *line_vals: List\[Any\]*, *line_names: List\[str\]*, *plot_name: str*, *args: Dict\[str, Any\]*, *xlabel: str = \'\'*, *ylabel: str = \'\'*, *x_log: bool = False*, *y_log: bool = False*, *styles=None*)¶

    :   Constructor. x_vals can be a list of scalars or a list of tuples/lists. If x_vals is a list of scalars and there are multiple x_names, all arguments will have the same value. If x_vals is a list of tuples/lists, each element should have the same length as x_names.

        Parameters:

        :   -   **x_names** (*List\[str\]*) -- Name of the arguments that should appear on the x axis of the plot.

            -   **x_vals** (*List\[Any\]*) -- List of values to use for the arguments in `x_names`.

            -   **line_arg** (*str*) -- Argument name for which different values correspond to different lines in the plot.

            -   **line_vals** (*List\[Any\]*) -- List of values to use for the arguments in `line_arg`.

            -   **line_names** (*List\[str\]*) -- Label names for the different lines.

            -   **plot_name** (*str*) -- Name of the plot.

            -   **args** (*Dict\[str,* *Any\]*) -- Dictionary of keyword arguments to remain fixed throughout the benchmark.

            -   **xlabel** (*str,* *optional*) -- Label for the x axis of the plot.

            -   **ylabel** (*str,* *optional*) -- Label for the y axis of the plot.

            -   **x_log** (*bool,* *optional*) -- Whether the x axis should be log scale.

            -   **y_log** (*bool,* *optional*) -- Whether the y axis should be log scale.

            -   **styles** (*list\[tuple\[str,* *str\]\]*) -- A list of tuples, where each tuple contains two elements: a color and a linestyle.

    Methods

      --------------------------------------------------- --------------
      `__init__`(self, x_names, x_vals, line_arg, \...)   Constructor.
      --------------------------------------------------- --------------

Previous Next

------------------------------------------------------------------------

© Copyright 2020, Philippe Tillet.

Built with Sphinx using a theme provided by Read the Docs.
