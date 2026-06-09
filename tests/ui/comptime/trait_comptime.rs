#![feature(rustc_attrs)]

trait Foo {
    #[rustc_comptime]
    //~^ ERROR: cannot be used on required trait methods
    const fn foo();
    //~^ ERROR: functions in traits cannot be declared const

    #[rustc_comptime]
    //~^ ERROR: cannot be used on provided trait methods
    const fn bar() {}
    //~^ ERROR: functions in traits cannot be declared const
}

struct Bar;

impl Bar {
    #[rustc_comptime]
    const fn foo() {}
}

impl Foo for Bar {
    #[rustc_comptime]
    //~^ ERROR: cannot be used on trait methods
    const fn foo() {}
    //~^ ERROR: functions in trait impls cannot be declared const
}

fn main() {}
