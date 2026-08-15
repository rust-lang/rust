// Regression test for issue #137668 where an indirect argument have been marked as nocapture
// despite the fact that callee did in fact capture the address.
//
//@ run-pass
//@ compile-flags: -Copt-level=2

#[inline(never)]
pub fn f(a: [u32; 64], b: [u32; 64]) -> bool {
    &a as *const _ as usize != &b as *const _ as usize
}

fn main() {
    static S: [u32; 64] = [0; 64];
    assert!(f(S, S));
}
