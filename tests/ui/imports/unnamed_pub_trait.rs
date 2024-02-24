//@ aux-build:unnamed_pub_trait_source.rs

/*
 * This crate declares an unnameable public path for our item. Make sure we don't suggest
 * importing it by name, and instead we suggest importing it by glob.
 */
extern crate unnamed_pub_trait_source;
//~^ HELP trait `Tr` which provides `method` is implemented but not in scope; perhaps you want to import it
//~| SUGGESTION unnamed_pub_trait_source::prelude::*; // trait Tr

fn main() {
    use unnamed_pub_trait_source::S;
    S.method();
    //~^ ERROR no method named `method` found for struct `S` in the current scope [E0599]
    //~| HELP items from traits can only be used if the trait is in scope
}
