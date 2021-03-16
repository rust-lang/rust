#![feature(const_impl_trait, const_fn_fn_ptr_basics, rustc_attrs)]
// revisions: min_tait full_tait
#![feature(min_type_alias_impl_trait)]
#![cfg_attr(full_tait, feature(impl_trait_in_bindings, type_alias_impl_trait))]
//[full_tait]~^ WARN incomplete
//[full_tait]~| WARN incomplete

type Foo = impl Fn() -> usize;
const fn bar() -> Foo { || 0usize }
const BAZR: Foo = bar();
//[min_tait]~^ ERROR not permitted here

#[rustc_error]
fn main() {} //[full_tait]~ ERROR
