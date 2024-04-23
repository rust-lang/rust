# `coverage-options`

This option controls details of the coverage instrumentation performed by
`-C instrument-coverage`.

Multiple options can be passed, separated by commas. Valid options are:

- `no-branch`, `branch` or `mcdc`: `branch` enables branch coverage instrumentation and `mcdc` further enables modified condition/decision coverage instrumentation. `no-branch` disables branch coverage instrumentation as well as mcdc instrumentation, which is same as do not pass `branch` or `mcdc`.
