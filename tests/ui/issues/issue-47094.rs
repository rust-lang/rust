#[repr(C, u8)] //~ ERROR conflicting representation hints
//~^ WARN this was previously accepted
enum Foo {
    A,
    B,
}

#[repr(C)] //~ ERROR conflicting representation hints
//~^ WARN this was previously accepted
#[repr(u8)]
enum Bar {
    A,
    B,
}

fn main() {}
