// revisions: min_tait full_tait
#![feature(min_type_alias_impl_trait)]
#![cfg_attr(full_tait, feature(type_alias_impl_trait))]
//[full_tait]~^ WARN incomplete

fn main() {
}

type PartiallyDefined<T> = impl 'static;
//~^ ERROR: at least one trait must be specified

fn partially_defined<T: std::fmt::Debug>(_: T) -> PartiallyDefined<T> {
    4u32
}

type PartiallyDefined2<T> = impl 'static;
//~^ ERROR: at least one trait must be specified

fn partially_defined2<T: std::fmt::Debug>(_: T) -> PartiallyDefined2<T> {
    4u32
}

fn partially_defined22<T>(_: T) -> PartiallyDefined2<T> {
    4u32
}
