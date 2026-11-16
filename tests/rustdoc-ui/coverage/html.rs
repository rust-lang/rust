//@ compile-flags:-Z unstable-options --output-format html --coverage

/// Foo
pub struct Xo;

//~? ERROR `--output-format=html` is not supported for the `--coverage` option
