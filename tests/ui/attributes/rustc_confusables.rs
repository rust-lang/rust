//@ aux-build: rustc_confusables_across_crate.rs

#![feature(rustc_attrs)]

extern crate rustc_confusables_across_crate;

use rustc_confusables_across_crate::BTreeSet;

fn main() {
    // Misspellings (similarly named methods) take precedence over `rustc_confusables`.
    let x = BTreeSet {};
    x.inser();
    //~^ ERROR no method named
    //~| HELP there is a method `insert` with a similar name
    x.foo();
    //~^ ERROR no method named
    x.push();
    //~^ ERROR no method named
    //~| HELP you might have meant to use `insert`
    x.test();
    //~^ ERROR no method named
    x.pulled();
    //~^ ERROR no method named
    //~| HELP you might have meant to use `pull`
}

struct Bar;

impl Bar {
    #[rustc_confusables()]
    //~^ ERROR expected at least one confusable name
    fn baz() {}

    #[rustc_confusables]
    //~^ ERROR malformed `rustc_confusables` attribute input
    //~| HELP must be of the form
    fn qux() {}

    #[rustc_confusables(invalid_meta_item)]
    //~^ ERROR malformed `rustc_confusables` attribute input [E0539]
    //~| HELP must be of the form
    fn quux() {}
}

#[rustc_confusables("blah")]
//~^ ERROR attribute cannot be used on
//~| HELP can only be applied to
//~| HELP remove the attribute
fn not_inherent_impl_method() {}
