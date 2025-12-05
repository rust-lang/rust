//@ revisions: normal import_trait_associated_functions
//@[import_trait_associated_functions] check-pass
#![cfg_attr(import_trait_associated_functions, feature(import_trait_associated_functions))]

// Makes sure that imported associated functions are shadowed by the local declarations.

use A::b; //[normal]~ ERROR `use` associated items of traits is unstable

trait A {
    fn b() {}
}

fn main() {
    let b: ();
}
