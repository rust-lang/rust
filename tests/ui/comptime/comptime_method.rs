#![feature(comptime)]

struct Bar;

#[comptime]
impl Bar {
    fn boo(&self) {}
}

const _: () = {
    Bar.boo();
};

fn main() {
    Bar.boo();
    //~^ ERROR: comptime functions can only be called at compile time
}
