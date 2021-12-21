// aux-build:overlapping_pub_trait_source.rs

/*
 * This crate declares two public paths, `m::Tr` and `prelude::_`. Make sure we prefer the former.
 */
extern crate overlapping_pub_trait_source;

fn main() {
    //~^ HELP the following trait is implemented but not in scope; perhaps add a `use` for it:
    //~| SUGGESTION overlapping_pub_trait_source::m::Tr
    use overlapping_pub_trait_source::S;
    S.method();
    //~^ ERROR no method named `method` found for struct `S` in the current scope [E0599]
    //~| HELP items from traits can only be used if the trait is in scope
}
