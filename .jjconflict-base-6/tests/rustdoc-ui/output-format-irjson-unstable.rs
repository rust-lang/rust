//@ compile-flags: --output-format=json
pub struct Foo;

//~? ERROR the -Z unstable-options flag must be passed to enable --output-format=json for documentation
