// check-pass

#[repr(C,u8)] //~ WARNING conflicting representation hints
enum Foo {
    A,
    B,
}

#[repr(C)] //~ WARNING conflicting representation hints
#[repr(u8)]
enum Bar {
    A,
    B,
}

fn main() {}
