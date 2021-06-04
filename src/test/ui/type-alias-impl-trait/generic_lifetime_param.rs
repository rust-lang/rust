// build-pass (FIXME(62277): could be check-pass?)

// revisions: min_tait full_tait
#![feature(min_type_alias_impl_trait)]
#![cfg_attr(full_tait, feature(type_alias_impl_trait))]
//[full_tait]~^ WARN incomplete

fn main() {}

type Region<'a> = impl std::fmt::Debug;

fn region<'b>(a: &'b ()) -> Region<'b> {
    a
}
