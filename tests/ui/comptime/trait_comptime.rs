#![feature(rustc_attrs)]

trait Foo {
    #[rustc_comptime]
    //~^ ERROR: cannot be used on required trait methods
    fn foo();

    #[rustc_comptime]
    //~^ ERROR: cannot be used on provided trait methods
    fn bar() {}
}

struct Bar;

impl Bar {
    #[rustc_comptime]
    fn foo() {}
}

impl Foo for Bar {
    #[rustc_comptime]
    //~^ ERROR: cannot be used on trait methods
    fn foo() {}
}

fn main() {}
