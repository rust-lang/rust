//! Regression test for issue #1399
//!
//! This tests that the compiler's last-use analysis correctly handles variables
//! that are captured by closures (upvars). The original issue was that the analysis
//! would incorrectly optimize variable usage as "last use" and perform moves, even when
//! the variable was later needed by a closure that captured it.
//!
//! See: https://github.com/rust-lang/rust/issues/1399

//@ run-pass

struct A {
    _a: Box<isize>,
}

fn foo() -> Box<dyn FnMut() -> isize + 'static> {
    let k: Box<_> = Box::new(22);

    // This use of k.clone() should not be treated as a "last use"
    // even though the closure below doesn't actually capture k
    let _u = A { _a: k.clone() };

    // The closure doesn't actually use k, but the analyzer needs to handle
    // the potential capture scenario correctly
    let result = || 22;

    Box::new(result)
}

pub fn main() {
    assert_eq!(foo()(), 22);
}
