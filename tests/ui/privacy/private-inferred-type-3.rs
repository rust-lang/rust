// aux-build:private-inferred-type.rs

// error-pattern:type `{fn item ext::priv_fn: fn()}` is private
// error-pattern:static `ext::PRIV_STATIC` is private
// error-pattern:type `ext::PrivEnum` is private
// error-pattern:type `{fn item <u8 as ext::PrivTrait>::method: fn()}` is private
// error-pattern:type `{fn item <u8 as ext::PrivTrait>::method: fn()}` is private
// error-pattern:type `{fn item ext::PrivTupleStruct: fn(u8) -> ext::PrivTupleStruct}` is private
// error-pattern:type `{fn item Pub::<u8>::priv_method: for<'a> fn(&'a Pub<u8>)}` is private

#![feature(decl_macro)]

extern crate private_inferred_type as ext;

fn main() {
    ext::m!();
}
