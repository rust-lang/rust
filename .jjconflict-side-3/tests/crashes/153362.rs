//@ known-bug: #153362
struct ThinDst {
    b: unsafe<> (),
}

const C1: &ThinDst = unsafe { std::mem::transmute(b"d".as_ptr()) };
