//@ run-pass
//@ compile-flags: -Copt-level=0 -Cdebuginfo=2

// Make sure LLVM does not miscompile this.

fn indirect_get_slice() -> &'static [usize] {
    &[]
}

#[inline(always)]
fn get_slice() -> &'static [usize] {
    let ret = indirect_get_slice();
    ret
}

fn main() {
    let output = get_slice().len();
    assert_eq!(output, 0);
}
