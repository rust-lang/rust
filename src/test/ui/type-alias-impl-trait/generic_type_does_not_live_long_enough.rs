// revisions: min_tait full_tait
#![feature(min_type_alias_impl_trait)]
#![cfg_attr(full_tait, feature(type_alias_impl_trait))]
//[full_tait]~^ WARN incomplete

fn main() {
    let y = 42;
    let x = wrong_generic(&y);
    let z: i32 = x; //~ ERROR mismatched types
}

type WrongGeneric<T> = impl 'static;
//~^ ERROR the parameter type `T` may not live long enough
//~| ERROR the parameter type `T` may not live long enough
//~| ERROR: at least one trait must be specified

fn wrong_generic<T>(t: T) -> WrongGeneric<T> {
    t
}
