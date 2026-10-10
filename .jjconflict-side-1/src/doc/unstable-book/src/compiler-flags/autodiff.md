# `autodiff`

The tracking issue for this feature is: [#124509](https://github.com/rust-lang/rust/issues/124509).

------------------------

This feature allows you to differentiate functions using automatic differentiation.
Set the `-Zautodiff=<options>` compiler flag to adjust the behaviour of the autodiff feature.
Multiple options can be separated with a comma.

## Syntax
```bash
rustc -Z autodiff=Enable[,options]
```

Where `options` can be:

- `Enable` - Required flag to enable autodiff
- `PrintTA` - print Type Analysis Information
- `PrintTAFn=<fn_name>` - print Type Analysis Information for a specific function (consider combining it with `no_mangle`)
- `PrintAA` - print Activity Analysis Information
- `PrintPerf` - print Performance Warnings from Enzyme
- `PrintSteps` - prints all intermediate transformations
- `PrintModBefore` - print the whole module, before running opts
- `PrintModAfter` - print the module after Enzyme differentiated everything
- `LooseTypes` - Enzyme's loose type debug helper (can cause incorrect gradients)
- `Inline` - runs Enzyme specific Inlining
- `RuntimeActivity` - allow specifying activity at runtime


## Examples

```bash
# Enable autodiff via cargo, assuming `enzyme` being a toolchain that supports autodiff
"RUSTFLAGS=-Zautodiff=Enable" cargo +enzyme build

# Enable autodiff directly via rustc
rustc -Z autodiff=Enable

# Print TypeAnalysis updates for the function `foo`, as well as Activity Analysis for all differentiated code.
rustc -Z autodiff=Enable,PrintTAFn=foo,PrintAA
```
