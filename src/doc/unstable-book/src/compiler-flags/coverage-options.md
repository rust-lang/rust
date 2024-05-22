# `coverage-options`

This option controls details of the coverage instrumentation performed by
`-C instrument-coverage`.

Multiple options can be passed, separated by commas. Valid options are:

- `block`, `branch`, `mcdc`:
  Sets the level of coverage instrumentation.
  Setting the level will override any previously-specified level.
  - `block` (default):
    Blocks in the control-flow graph will be instrumented for coverage.
  - `branch`:
    In addition to block coverage, also enables branch coverage instrumentation.
  - `mcdc`:
    In addition to block and branch coverage, also enables MC/DC instrumentation.
    (Branch coverage instrumentation may differ in some cases.)
