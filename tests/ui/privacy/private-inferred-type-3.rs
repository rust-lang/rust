//@ aux-build:private-inferred-type.rs

#![feature(decl_macro)]

extern crate private_inferred_type as ext;

fn main() {
    ext::m!();
    //~^ ERROR type `fn() {ext::priv_fn}` is private
    //~| ERROR static `ext::PRIV_STATIC` is private
    //~| ERROR type `ext::PrivEnum` is private
    //~| ERROR type `fn() {<u8 as ext::PrivTrait>::method}` is private
    //~| ERROR type `fn(u8) -> ext::PrivTupleStruct {ext::PrivTupleStruct}` is private
    //~| ERROR type `fn(u8) -> PubTupleStruct {PubTupleStruct}` is private
    //~| ERROR type `for<'a> fn(&'a Pub<u8>) {Pub::<u8>::priv_method}` is private
}
