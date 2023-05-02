#[derive(Copy, Clone)]
pub struct Foo {
    x: [u8; SIZE],
    //~^ ERROR the constant `1` is not of type `usize`
}

const SIZE: u32 = 1;

fn main() {}
