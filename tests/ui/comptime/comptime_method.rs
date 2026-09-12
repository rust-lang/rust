#![feature(rustc_attrs)]

struct Bar;

#[rustc_comptime]
impl Bar {
    fn boo(&self) {}
}

const _: () = {
    Bar.boo();
};

fn main() {
    Bar.boo();
    //~^ ERROR: comptime fns can only be called at compile time
}
