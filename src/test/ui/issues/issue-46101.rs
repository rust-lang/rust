#![feature(use_extern_macros)]
trait Foo {}
#[derive(Foo::Anything)] //~ ERROR failed to resolve: partially resolved path in a derive macro
struct S;
