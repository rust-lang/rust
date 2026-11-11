//@ revisions: current next
//@ ignore-compare-mode-next-solver (explicit revisions)
//@[next] compile-flags: -Znext-solver
//@[next] check-pass

#![feature(explicit_tail_calls)]
#![expect(incomplete_features)]

// Regression test for https://github.com/rust-lang/rust/issues/139305.
//
// Combining return position impl trait (RPIT) with guaranteed tail calls does not
// currently work, but at least it does not ICE.

fn foo(x: u32, y: u32) -> u32 {
    x + y
}

fn bar(x: u32, y: u32) -> impl ToString {
    become foo(x, y);
    //[current]~^ ERROR mismatched signatures
}

fn main() {
    assert_eq!(bar(1, 2).to_string(), "3");
}
