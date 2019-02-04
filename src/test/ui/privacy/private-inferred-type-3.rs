// aux-build:private-inferred-type.rs

// error-pattern:function `ext::priv_fn` is private
// error-pattern:static `PRIV_STATIC` is private
// error-pattern:enum `ext::PrivEnum` is private
// error-pattern:method `ext::PrivTrait::method` is private
// error-pattern:struct `ext::PrivTupleStruct` is private
// error-pattern:struct `ext::PubTupleStruct` is private
// error-pattern:method `<ext::Pub<u8>>::priv_method` is private

#![feature(decl_macro)]

extern crate private_inferred_type as ext;

fn main() {
    ext::m!();
}
