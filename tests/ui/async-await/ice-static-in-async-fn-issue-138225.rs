//! Regression test for <https://github.com/rust-lang/rust/issues/138225>
//!
//! This used to ICE with:
//! `called Result::unwrap() on an Err value: ReferencesError(ErrorGuaranteed(()))`
//! when compiling with optimizations enabled (`-C opt-level=1` or higher).
//!
//! The bug occurred when using an undefined type in a struct field combined with
//! a static reference inside an async function.

//@ edition: 2021

pub struct A {
    name: NestedOption<Option<String>>,
    //~^ ERROR cannot find type `NestedOption` in this scope
}

impl A {
    pub async fn func1() -> &'static A {
        static RES: A = A { name: None };
        &RES
    }
}

fn main() {}
