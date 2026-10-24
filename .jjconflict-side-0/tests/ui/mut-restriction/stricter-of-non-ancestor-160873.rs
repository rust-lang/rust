//! Issue: <https://github.com/rust-lang/rust/issues/160873>
//! This test checks that restricting struct expressions with
//! non-ancestor mutability restrictions do not cause an ICE.

//@ edition: 2018..
#![feature(mut_restriction)]

pub mod inner {
    pub struct InnerS {
        pub mut(self) x: i32,
        pub mut(in std) y: i32, //~ ERROR field mutation can only be restricted to ancestor modules
    }
}

fn main() {
    let _ = inner::InnerS { x: 0, y: 0 }; //~ ERROR `InnerS` cannot be constructed using a `struct` expression outside `crate::inner`
}
