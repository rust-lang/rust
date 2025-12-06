// Check some places in which we want to split multi-character punctuation.
//@ edition: 2015
//@ check-pass
#![feature(never_type)] // only used inside `bang_eq_never_ty`!
#![expect(bare_trait_objects)]

use std::fmt::Debug;

fn plus_eq_bound() {
    // issue: <https://github.com/rust-lang/rust/issues/47856>
    struct W<T: Clone + = ()> { t: T }
    struct S<T: Clone += ()> { t: T }

    // Bare & `dyn`-prefixed trait object types take different paths in the parser.
    // Therefore, test both branches.

    let _: Debug + = *(&() as &dyn Debug);
    let _: Debug += *(&() as &dyn Debug);

    let _: dyn Debug + = *(&() as &dyn Debug);
    let _: dyn Debug += *(&() as &dyn Debug);

    #[cfg(false)] fn w() where Trait + = () {}
    #[cfg(false)] fn s() where Trait += () {}
}

fn bang_eq_never_ty(x: !) {
    let _: ! = x;
    let _: != x;

    #[cfg(false)] struct W<const X: ! = { loop {} }>;
    #[cfg(false)] struct S<const X: != { loop {} }>;
}

fn main() {}
