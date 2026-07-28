#![feature(comptime)]

trait Foo {
    #[comptime]
    //~^ ERROR: functions in traits cannot be declared #[comptime]
    fn foo();

    #[comptime]
    //~^ ERROR: functions in traits cannot be declared #[comptime]
    fn bar() {}
}

struct Bar;

impl Bar {
    #[comptime]
    fn foo() {}
}

impl Foo for Bar {
    #[comptime]
    //~^ ERROR: functions in trait impls cannot be declared #[comptime]
    fn foo() {}
}

fn main() {}
