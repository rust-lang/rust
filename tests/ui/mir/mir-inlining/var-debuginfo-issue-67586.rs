//@ run-pass
//@ compile-flags: -Z mir-opt-level=3 -C opt-level=0 -C debuginfo=2

#[inline(never)]
pub fn foo(bar: usize) -> usize {
    std::convert::identity(bar)
}

fn main() {
    foo(0);
}
