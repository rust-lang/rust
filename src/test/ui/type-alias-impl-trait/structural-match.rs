#![feature(const_impl_trait)]
// revisions: min_tait full_tait
#![feature(min_type_alias_impl_trait)]
#![cfg_attr(full_tait, feature(type_alias_impl_trait, impl_trait_in_bindings))]
//[full_tait]~^ WARN incomplete
//[full_tait]~| WARN incomplete

type Foo = impl Send;

// This is not structural-match
struct A;

const fn value() -> Foo {
    A
}
const VALUE: Foo = value(); //[min_tait]~ ERROR not permitted here

fn test() {
    match todo!() {
        VALUE => (),
        //[full_tait]~^ `impl Send` cannot be used in patterns
        _ => (),
    }
}

fn main() {}
