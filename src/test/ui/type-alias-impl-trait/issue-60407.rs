// check-pass

// revisions: min_tait full_tait
#![feature(min_type_alias_impl_trait)]
#![cfg_attr(full_tait, feature(type_alias_impl_trait, impl_trait_in_bindings))]
//[full_tait]~^ WARN incomplete
//[full_tait]~| WARN incomplete

type Debuggable = impl core::fmt::Debug;

static mut TEST: Option<Debuggable> = None;

fn main() {
    unsafe { TEST = Some(foo()) }
}

fn foo() -> Debuggable {
    0u32
}
