// ICE in mir building with captured value of unresolved type
// None in compiler/rustc_mir_build/src/build/expr/as_place.rs
// issue: rust-lang/rust#110453
//@ edition:2021

#![crate_type="lib"]

pub fn dup(f: impl Fn(i32) -> i32) -> impl Fn(as_str) -> i32 {
//~^ ERROR cannot find type `as_str` in this scope
    move |a| f(a * 2)
}
