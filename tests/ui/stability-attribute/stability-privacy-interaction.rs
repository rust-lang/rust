//! Regression test for issue #38412: interaction between stability attributes and privacy
//!
//! Tests that the compiler correctly handles the interaction between feature gates
//! and privacy/visibility rules. Specifically verifies that enabled unstable features
//! are accessible while disabled ones are properly rejected.

//@ aux-build:pub-and-stability.rs

// Enable `unstable_declared` but not `unstable_undeclared` to test
// that the compiler allows enabled features but rejects disabled ones
#![feature(unstable_declared)]

extern crate pub_and_stability;
use pub_and_stability::{Record, Trait, Tuple};

fn main() {
    // Test struct field access patterns
    let Record { .. } = Record::new();

    let Record {
        a_stable_pub: _,
        a_unstable_declared_pub: _,
        ..
    } = Record::new();

    let Record {
        a_stable_pub: _,
        a_unstable_declared_pub: _,
        a_unstable_undeclared_pub: _,  //~ ERROR use of unstable library feature `unstable_undeclared`
        ..
    } = Record::new();

    let r = Record::new();
    let t = Tuple::new();

    // Test field access with different stability/privacy combinations
    r.a_stable_pub;
    r.a_unstable_declared_pub;
    r.a_unstable_undeclared_pub; //~ ERROR use of unstable library feature
    r.b_crate;                   //~ ERROR is private
    r.c_mod;                     //~ ERROR is private
    r.d_priv;                    //~ ERROR is private

    t.0;
    t.1;
    t.2;                         //~ ERROR use of unstable library feature
    t.3;                         //~ ERROR is private
    t.4;                         //~ ERROR is private
    t.5;                         //~ ERROR is private

    // Test trait method access
    r.stable_trait_method();
    r.unstable_declared_trait_method();
    r.unstable_undeclared_trait_method(); //~ ERROR use of unstable library feature

    // Test inherent method access
    r.stable();
    r.unstable_declared();
    r.unstable_undeclared();              //~ ERROR use of unstable library feature

    r.pub_crate();                        //~ ERROR `pub_crate` is private
    r.pub_mod();                          //~ ERROR `pub_mod` is private
    r.private();                          //~ ERROR `private` is private

    // Repeat tests for tuple struct
    let t = Tuple::new();
    t.stable_trait_method();
    t.unstable_declared_trait_method();
    t.unstable_undeclared_trait_method(); //~ ERROR use of unstable library feature

    t.stable();
    t.unstable_declared();
    t.unstable_undeclared();              //~ ERROR use of unstable library feature

    t.pub_crate();                        //~ ERROR `pub_crate` is private
    t.pub_mod();                          //~ ERROR `pub_mod` is private
    t.private();                          //~ ERROR `private` is private
}
