# `coverage-options`

This option controls details of the coverage instrumentation performed by
`-C instrument-coverage`.

Multiple options can be passed, separated by commas. Valid options are:

- `block`, `branch`, `condition`:
  Sets the level of coverage instrumentation.
  Setting the level will override any previously-specified level.
  - `block` (default):
    Blocks in the control-flow graph will be instrumented for coverage.
  - `branch`:
    In addition to block coverage, also enables branch coverage instrumentation.
  - `condition`:
    In addition to branch coverage, also instruments some boolean expressions
    as branches, even if they are not directly used as branch conditions.
