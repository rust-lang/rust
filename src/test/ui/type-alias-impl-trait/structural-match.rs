#![feature(const_impl_trait)]
// revisions: min_tait full_tait
#![cfg_attr(min_tait, feature(min_type_alias_impl_trait))]
#![cfg_attr(not(min_tait), feature(type_alias_impl_trait, min_type_alias_impl_trait))]
//[full_tait]~^ WARN incomplete

type Foo = impl Send;

// This is not structural-match
struct A;

const fn value() -> Foo {
    A
}
const VALUE: Foo = value();

fn test() {
    match todo!() {
        VALUE => (),
        //~^ `impl Send` cannot be used in patterns
        _ => (),
    }
}

fn main() {}
