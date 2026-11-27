//@ check-pass
#[derive(Copy, Clone)]
pub struct Foo { a: bool }

pub union Bar {
    a: Foo,
    b: u32,
}
pub fn baz(mut bar: Bar) {
    unsafe {
        { bar.a }.a = true;
    }
}

fn main() {}
