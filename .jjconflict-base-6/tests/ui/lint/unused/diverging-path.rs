//! Assignments to a captured variable within a diverging closure should not be considered unused if
//! the divergence is caught.
//!
//! Regression test for https://github.com/rust-lang/rust/issues/152079
//@ compile-flags: -Wunused
//@ check-pass

fn main() {
    let mut x = 1;
    catch(|| {
        x = 2;
        panic!();
    });
    dbg!(x);
}

fn catch<F: FnOnce()>(f: F) {
    if let Ok(true) = std::fs::exists("may_or_may_not_call_f") {
        _ = std::panic::catch_unwind(std::panic::AssertUnwindSafe(f));
    }
}
