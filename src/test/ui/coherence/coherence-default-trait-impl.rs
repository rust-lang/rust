#![feature(optin_builtin_traits)]

auto trait MySafeTrait {}

struct Foo;

unsafe impl MySafeTrait for Foo {}
//~^ ERROR E0199

unsafe auto trait MyUnsafeTrait {}

impl MyUnsafeTrait for Foo {}
//~^ ERROR E0200

fn main() {}
