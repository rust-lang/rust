# `autodiff`

The tracking issue for this feature is: [#124509](https://github.com/rust-lang/rust/issues/124509).

------------------------

This feature allows you to differentiate functions using automatic differentiation.
Set the `-Zautodiff=<options>` compiler flag to adjust the behaviour of the autodiff feature.
Multiple options can be separated with a comma. Valid options are:

`Enable` - Required flag to enable autodiff
`PrintTA` - print Type Analysis Information
`PrintTAFn` - print Type Analysis Information for a specific function
`PrintAA` - print Activity Analysis Information
`PrintPerf` - print Performance Warnings from Enzyme
`PrintSteps` - prints all intermediate transformations
`PrintModBefore` - print the whole module, before running opts
`PrintModAfter` - print the module after Enzyme differentiated everything
`LooseTypes` - Enzyme's loose type debug helper (can cause incorrect gradients)
`Inline` - runs Enzyme specific Inlining
`RuntimeActivity` - allow specifying activity at runtime
