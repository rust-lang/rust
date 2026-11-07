// This used to ICE during codegen after MIR inlining of g into f.
// The root cause was a missing fold of length constant in Rvalue::Repeat.
// Regression test for #76248.
//
//@ build-pass
//@ compile-flags: -Zmir-opt-level=3

const N: usize = 1;

pub struct Elem<M> {
    pub x: [usize; N],
    pub m: M,
}

pub fn f() -> Elem<()> {
    g(())
}

#[inline]
pub fn g<M>(m: M) -> Elem<M> {
    Elem {
        x: [0; N],
        m,
    }
}

pub fn main() {
    f();
}
