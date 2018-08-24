#![feature(optin_builtin_traits)]

auto trait MySafeTrait {}

struct Foo;

unsafe impl MySafeTrait for Foo {}
//~^ ERROR implementing the trait `MySafeTrait` is not unsafe

unsafe auto trait MyUnsafeTrait {}

impl MyUnsafeTrait for Foo {}
//~^ ERROR the trait `MyUnsafeTrait` requires an `unsafe impl` declaration

fn main() {}
