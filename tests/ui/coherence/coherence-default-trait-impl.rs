#![feature(rustc_attrs)]
#![feature(negative_impls)]

#[rustc_auto_trait]
trait MySafeTrait {}

struct Foo;

unsafe impl MySafeTrait for Foo {}
//~^ ERROR E0199

#[rustc_auto_trait]
unsafe trait MyUnsafeTrait {}

impl MyUnsafeTrait for Foo {}
//~^ ERROR E0200

fn main() {}
