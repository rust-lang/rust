// ICE in mir building with captured value of unresolved type
// None in compiler/rustc_mir_build/src/build/expr/as_place.rs
// issue: rust-lang/rust#110453
//@ edition:2021

#![crate_type="lib"]

pub struct B;
pub fn a() -> B { B }

mod handlers {
    pub struct C(B);
    //~^ ERROR cannot find type `B` in this scope
    pub fn c() -> impl Fn() -> C {
        let a1 = ();
        || C((crate::a(), a1).into())
    }
}
