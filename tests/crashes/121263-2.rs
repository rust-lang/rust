//@ known-bug: #121263
#[repr(C)]
#[derive(Debug)]
struct L {
    _: MyI32,
}
