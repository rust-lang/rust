// build-pass (FIXME(62277): could be check-pass?)

#[repr(C,u8)]
enum Foo {
    A,
    B,
}

#[repr(C)]
#[repr(u8)]
enum Bar {
    A,
    B,
}

fn main() {}
