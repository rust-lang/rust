# Continuous Integration

It is recommended to run Clippy on CI with `-Dwarnings`, so that Clippy lints
prevent CI from passing. To enforce errors on warnings on all `cargo` commands
not just `cargo clippy`, you can set the env var `RUSTFLAGS="-Dwarnings"`.

We recommend to use Clippy from the same toolchain, that you use for compiling
your crate for maximum compatibility. E.g. if your crate is compiled with the
`stable` toolchain, you should also use `stable` Clippy.

> _Note:_ New Clippy lints are first added to the `nightly` toolchain. If you
> want to help with improving Clippy and have CI resources left, please consider
> adding a `nightly` Clippy check to your CI and report problems like false
> positives back to us. With that we can fix bugs early, before they can get to
> stable.

This chapter will give an overview on how to use Clippy on different popular CI
providers.
