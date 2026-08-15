//@ compile-flags:-Z unstable-options --output-format html --show-coverage

/// Foo
pub struct Xo;

//~? ERROR `--output-format=html` is not supported for the `--show-coverage` option
