// revisions: min_tait full_tait
#![feature(min_type_alias_impl_trait)]
#![cfg_attr(full_tait, feature(type_alias_impl_trait))]
//[full_tait]~^ WARN incomplete

fn main() {}

type Two<'a, 'b> = impl std::fmt::Debug;

fn one<'a>(t: &'a ()) -> Two<'a, 'a> { //~ ERROR non-defining opaque type use
    t
}
