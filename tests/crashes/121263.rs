//@ known-bug: #121263
#[repr(C)]
#[repr(C)]
#[derive(Debug)]
struct L {
    _: i32,
    _: MyI32,
    _: BadEnum,
}
