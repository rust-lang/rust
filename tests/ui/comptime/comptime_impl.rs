#![feature(rustc_attrs, const_trait_impl)]

const trait Foo {
    fn foo(&self);

    fn bar(&self) {}
}

struct Bar;

#[rustc_comptime]
impl Bar {
    fn boo(&self) {}
}

#[rustc_comptime]
//~^ ERROR: cannot be used on trait impl
impl Foo for Bar {
    fn foo(&self) {
        comptime_fn();
    }
}

#[rustc_comptime]
fn comptime_fn() {}

const _: () = {
    Bar.boo();
    Bar.foo();
    //~^ ERROR: `Bar: const Foo` is not satisfied
    Bar.bar();
    //~^ ERROR: `Bar: const Foo` is not satisfied
};

fn main() {
    // FIXME(comptime): this should not be allowed, as the impl is comptime
    Bar.foo();
}
