//! regression test for <https://github.com/rust-lang/rust/issues/23477>
//@ build-pass
//@ revisions: current next
//@ ignore-compare-mode-next-solver (explicit revisions)
//@ [next] compile-flags: -Znext-solver
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
