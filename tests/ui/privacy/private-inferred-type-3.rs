//@ aux-build:private-inferred-type.rs

//@ error-pattern:type `fn() {ext::priv_fn}` is private
//@ error-pattern:static `ext::PRIV_STATIC` is private
//@ error-pattern:type `ext::PrivEnum` is private
//@ error-pattern:type `fn() {<u8 as ext::PrivTrait>::method}` is private
//@ error-pattern:type `fn(u8) -> ext::PrivTupleStruct {ext::PrivTupleStruct}` is private
//@ error-pattern:type `fn(u8) -> PubTupleStruct {PubTupleStruct}` is private
//@ error-pattern:type `for<'a> fn(&'a Pub<u8>) {Pub::<u8>::priv_method}` is private

#![feature(decl_macro)]

extern crate private_inferred_type as ext;

fn main() {
    ext::m!();
}
