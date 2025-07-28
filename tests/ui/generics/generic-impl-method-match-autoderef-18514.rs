//@ run-pass
// Test that we don't ICE when codegenning a generic impl method from
// an extern crate that contains a match expression on a local
// variable place where one of the match case bodies contains an
// expression that autoderefs through an overloaded generic deref
// impl.

//@ aux-build:generic-impl-method-match-autoderef-18514.rs

extern crate generic_impl_method_match_autoderef_18514 as ice;
use ice::{Tr, St};

fn main() {
    let st: St<()> = St(vec![]);
    st.tr();
}

// https://github.com/rust-lang/rust/issues/18514
