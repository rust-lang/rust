// revisions: min_tait full_tait
#![feature(min_type_alias_impl_trait, rustc_attrs)]
#![cfg_attr(full_tait, feature(type_alias_impl_trait, impl_trait_in_bindings))]
//[full_tait]~^ WARN incomplete
//[full_tait]~| WARN incomplete

type Debuggable = impl core::fmt::Debug;

static mut TEST: Option<Debuggable> = None; //[min_tait]~ ERROR not permitted here

#[rustc_error]
fn main() { //[full_tait]~ ERROR
    unsafe { TEST = Some(foo()) }
}

fn foo() -> Debuggable { //[min_tait]~ ERROR concrete type differs
    0u32
}
