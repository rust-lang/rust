//@ known-bug: #121722
#[repr(C)]
struct Foo {
    _: u8,
}

#[repr(C)]
struct D {
    _: Foo,
}
