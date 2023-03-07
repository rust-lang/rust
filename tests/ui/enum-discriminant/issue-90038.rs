// run-pass

#[repr(u32)]
pub enum Foo {
    // Greater than or equal to 2
    A = 2,
}

pub enum Bar {
    A(Foo),
    // More than two const variants
    B,
    C,
}

fn main() {
    match Bar::A(Foo::A) {
        Bar::A(_) => (),
        _ => unreachable!(),
    }
}
