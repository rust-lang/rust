//@ build-pass
//@ compile-flags: -g

pub struct Dst {
    pub a: (),
    pub b: (),
    pub data: [u8],
}

pub unsafe fn borrow(bytes: &[u8]) -> &Dst {
    let dst: &Dst = std::mem::transmute((bytes.as_ptr(), bytes.len()));
    dst
}

fn main() {}
