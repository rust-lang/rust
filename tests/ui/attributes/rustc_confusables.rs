// aux-build: rustc_confusables_across_crate.rs

#![feature(rustc_attrs)]

extern crate rustc_confusables_across_crate;

use rustc_confusables_across_crate::BTreeSet;

fn main() {
    // Misspellings (similarly named methods) take precedence over `rustc_confusables`.
    let x = BTreeSet {};
    x.inser();
    //~^ ERROR no method named
    //~| HELP there is a method with a similar name
    x.foo();
    //~^ ERROR no method named
    x.push();
    //~^ ERROR no method named
    //~| HELP you might have meant to use `insert`
    x.test();
    //~^ ERROR no method named
    x.pulled();
    //~^ ERROR no method named
    //~| HELP there is a method with a similar name
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
    //~^ ERROR expected a quoted string literal
    //~| HELP consider surrounding this with quotes
    fn quux() {}
}

#[rustc_confusables("blah")]
//~^ ERROR attribute should be applied to an inherent method
fn not_inherent_impl_method() {}
