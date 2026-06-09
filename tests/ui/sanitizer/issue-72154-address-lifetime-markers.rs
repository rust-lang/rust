// Regression test for issue 72154, where the use of AddressSanitizer enabled
// emission of lifetime markers during codegen, while at the same time asking
// always inliner pass not to insert them.  This eventually lead to a
// miscompilation which was subsequently detected by AddressSanitizer as UB.
//
//@ needs-sanitizer-support
//@ needs-sanitizer-address
//@ ignore-cross-compile
//
//@ compile-flags: -Copt-level=0 -Zsanitizer=address -C unsafe-allow-abi-mismatch=sanitizer
//@ run-pass

pub struct Wrap {
    pub t: [usize; 1]
}

impl Wrap {
    #[inline(always)]
    pub fn new(t: [usize; 1]) -> Self {
        Wrap { t }
    }
}

#[inline(always)]
pub fn assume_init() -> [usize; 1] {
    [1234]
}

fn main() {
    let x: [usize; 1] = assume_init();
    Wrap::new(x);
}
