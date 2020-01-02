#[repr(C, u8)] //~ ERROR conflicting representation hints
enum Foo {
    A,
    B,
}

#[repr(C)] //~ ERROR conflicting representation hints
#[repr(u8)]
enum Bar {
    A,
    B,
}

fn main() {}
