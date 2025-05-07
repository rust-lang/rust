//! Regression test for <https://github.com/rust-lang/rust/issues/133272>, where a `ref Self` ctor
//! makes it possible to hit a `delayed_bug` that was converted into a `span_bug` in
//! <https://github.com/rust-lang/rust/pull/121208>, and hitting this reveals that we did not have
//! test coverage for this specific code pattern (heh) previously.
//!
//! # References
//!
//! - ICE bug report: <https://github.com/rust-lang/rust/issues/133272>.
//! - Previous PR to change `delayed_bug` -> `span_bug`:
//!   <https://github.com/rust-lang/rust/pull/121208>
#![crate_type = "lib"]

struct Foo;

impl Foo {
    fn fun() {
        let S { ref Self } = todo!();
        //~^ ERROR expected identifier, found keyword `Self`
        //~| ERROR cannot find struct, variant or union type `S` in this scope
    }
}
