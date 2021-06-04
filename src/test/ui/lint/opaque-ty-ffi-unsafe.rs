// revisions: min_tait full_tait
#![feature(min_type_alias_impl_trait)]
#![cfg_attr(full_tait, feature(type_alias_impl_trait))]
//[full_tait]~^ WARN incomplete
#![deny(improper_ctypes)]

type A = impl Fn();

pub fn ret_closure() -> A {
    || {}
}

extern "C" {
    pub fn a(_: A);
//~^ ERROR `extern` block uses type `impl Fn<()>`, which is not FFI-safe
}

fn main() {}
