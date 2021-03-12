// compile-flags: -Zsave-analysis

// revisions: min_tait full_tait
#![feature(min_type_alias_impl_trait, rustc_attrs)]
#![cfg_attr(full_tait, feature(type_alias_impl_trait, impl_trait_in_bindings))]
//[full_tait]~^ WARN incomplete
//[full_tait]~| WARN incomplete

type T = impl Sized;
// The concrete type referred by impl-trait-type-alias(`T`) is guaranteed
// to be the same as where it occurs, whereas `impl Trait`'s instance is location sensitive;
// so difference assertion should not be declared on impl-trait-type-alias's instances.
// for details, check RFC-2515:
// https://github.com/rust-lang/rfcs/blob/master/text/2515-type_alias_impl_trait.md

fn take(_: fn() -> T) {}

#[rustc_error]
fn main() { //[full_tait]~ ERROR fatal error triggered by #[rustc_error]
    take(|| {});
    //[min_tait]~^ ERROR not permitted here
    take(|| {});
    //[min_tait]~^ ERROR not permitted here
}
