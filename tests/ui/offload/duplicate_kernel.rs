//@ compile-flags: -Zunstable-options -Zoffload=Device -Clto=fat -Csymbol-mangling-version=v0 --crate-name collision_kernels_a
//@ build-fail
//@ needs-offload

// An offload kernel whose mangled symbol collides with another item in the
// same crate must be rejected, just like any other symbol collision.

#![feature(rustc_attrs, gpu_offload)]
#![allow(internal_features)]

#[allow(non_snake_case)]
#[no_mangle]
pub fn _RNvC19collision_kernels_a6kernel(_x: f32) {}

#[rustc_offload_kernel]
fn kernel(_x: f32) {}
//~^ ERROR symbol `_RNvC19collision_kernels_a6kernel` is already defined

fn main() {
    _RNvC19collision_kernels_a6kernel(0.0);
    core::offload::offload! { kernel = kernel, args = (0.0f32,) }
}
