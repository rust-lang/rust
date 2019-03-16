// aux-build:pub-and-stability.rs

#![feature(unused_feature)]

// A big point of this test is that we *declare* `unstable_declared`,
// but do *not* declare `unstable_undeclared`. This way we can check
// that the compiler is letting in uses of declared feature-gated
// stuff but still rejecting uses of undeclared feature-gated stuff.
#![feature(unstable_declared)]

extern crate pub_and_stability;
use pub_and_stability::{Record, Trait, Tuple};

fn main() {
    // Okay
    let Record { .. } = Record::new();

    // Okay
    let Record { a_stable_pub: _, a_unstable_declared_pub: _, .. } = Record::new();

    let Record { a_stable_pub: _, a_unstable_declared_pub: _, a_unstable_undeclared_pub: _, .. } =
        Record::new();
    //~^^ ERROR use of unstable library feature 'unstable_undeclared'

    let r = Record::new();
    let t = Tuple::new();

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

    r.stable_trait_method();
    r.unstable_declared_trait_method();
    r.unstable_undeclared_trait_method(); //~ ERROR use of unstable library feature

    r.stable();
    r.unstable_declared();
    r.unstable_undeclared();              //~ ERROR use of unstable library feature

    r.pub_crate();                        //~ ERROR `pub_crate` is private
    r.pub_mod();                          //~ ERROR `pub_mod` is private
    r.private();                          //~ ERROR `private` is private

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
