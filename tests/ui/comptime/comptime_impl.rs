#![feature(const_trait_impl, comptime)]

const trait Foo {
    fn foo(&self);

    fn bar(&self) {}
}

struct Bar;

#[comptime]
impl Bar {
    fn boo(&self) {}
}

#[comptime]
impl Foo for Bar {
    fn foo(&self) {
        comptime_fn();
    }
}

#[comptime]
fn comptime_fn() {}

const _: () = {
    Bar.boo();
    Bar.foo();
    //~^ ERROR: `Bar: const Foo` is not satisfied
    Bar.bar();
    //~^ ERROR: `Bar: const Foo` is not satisfied
};

fn main() {}
