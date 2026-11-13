//@ compile-flags: --output-format=doctest
pub struct Foo;

//~? ERROR the -Z unstable-options flag must be passed to enable --output-format=doctest (see https://github.com/rust-lang/rust/issues/134529)
