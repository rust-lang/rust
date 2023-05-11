#[derive(Copy, Clone)]
pub struct Foo {
    x: [u8; SIZE],
    //~^ ERROR mismatched types
}

const SIZE: u32 = 1;

fn main() {}
