#![feature(rustc_attrs)]

struct Bar;

#[rustc_comptime]
//~^ ERROR: cannot be used on inherent impl
impl Bar {
    fn boo(&self) {}
}

const _: () = {
    Bar.boo();
    //~^ ERROR: cannot call non-const method `Bar::boo` in constants
};

fn main() {}
