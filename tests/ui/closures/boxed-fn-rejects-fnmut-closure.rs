//! Regression test for <https://github.com/rust-lang/rust/issues/26046>.
//!
//! Check that coercing a closure to `Box<dyn Fn()>` rejects a closure that only
//! implements `FnMut` because it mutates a captured variable.

fn foo() -> Box<dyn Fn()> {
    let num = 5;

    let closure = || { //~ ERROR expected a closure that
        num += 1;
    };

    Box::new(closure)
}

fn main() {}
