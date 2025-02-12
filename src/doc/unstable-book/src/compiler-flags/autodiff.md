# `autodiff`

The tracking issue for this feature is: [#124509](https://github.com/rust-lang/rust/issues/124509).

------------------------

This feature allows you to differentiate functions using automatic differentiation.
Set the `-Zautodiff=<options>` compiler flag to adjust the behaviour of the autodiff feature.
Multiple options can be separated with a comma. Valid options are:

`PrintTA` - print Type Analysis Information
`PrintAA` - print Activity Analysis Information
`PrintPerf` - print Performance Warnings from Enzyme
`Print` - prints all intermediate transformations
`PrintModBefore` - print the whole module, before running opts
`PrintModAfterOpts` - print the whole module just before we pass it to Enzyme
`PrintModAfterEnzyme` - print the module after Enzyme differentiated everything
`LooseTypes` - Enzyme's loose type debug helper (can cause incorrect gradients)
`Inline` - runs Enzyme specific Inlining
`NoModOptAfter` - do not optimize the module after Enzyme is done
`EnableFncOpt` - tell Enzyme to run LLVM Opts on each function it generated
`NoVecUnroll` - do not unroll vectorized loops
`RuntimeActivity` - allow specifying activity at runtime
