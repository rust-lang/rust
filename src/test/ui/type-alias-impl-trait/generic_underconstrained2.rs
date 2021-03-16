// revisions: min_tait full_tait
#![feature(min_type_alias_impl_trait)]
#![cfg_attr(full_tait, feature(type_alias_impl_trait))]
//[full_tait]~^ WARN incomplete

fn main() {}

type Underconstrained<T: std::fmt::Debug> = impl 'static;
//~^ ERROR: at least one trait must be specified

// not a defining use, because it doesn't define *all* possible generics
fn underconstrained<U>(_: U) -> Underconstrained<U> {
    //~^ ERROR `U` doesn't implement `Debug`
    5u32
}

type Underconstrained2<T: std::fmt::Debug> = impl 'static;
//~^ ERROR: at least one trait must be specified

// not a defining use, because it doesn't define *all* possible generics
fn underconstrained2<U, V>(_: U, _: V) -> Underconstrained2<V> {
    //~^ ERROR `V` doesn't implement `Debug`
    5u32
}
