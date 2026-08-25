//@ compile-flags: -Zunstable-options -Zoffload=Device -Clto=fat -Csymbol-mangling-version=v0
//@ build-fail
//@ needs-offload

// A generic offload kernel that is never called from host code (and hence
// never monomorphized) cannot be discovered without a manifest: with
// `-Zoffload=Device` (no manifest path), the compiler relies on
// monomorphization to find kernels, so it must reject the kernel rather than
// silently emit no device code for it.

#![feature(rustc_attrs)]
#![allow(internal_features)]

#[rustc_offload_kernel]
fn kernel<T: Copy>(x: T) {}
//~^ ERROR generic offload kernel `kernel` is not instantiated

fn main() {}
