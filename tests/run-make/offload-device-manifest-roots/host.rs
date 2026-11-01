#![feature(gpu_offload, rustc_attrs)]
#![allow(internal_features, dead_code)]

extern crate dep;

// Launched by `main`, so the manifest lists it and the device pass compiles it, together with
// everything it reaches.
#[rustc_offload_kernel]
fn launched(x: &mut f32) {
    dep::helper(x);
}

// Never launched, so it is not in the manifest and the device pass drops it, despite attribute.
#[rustc_offload_kernel]
fn dormant(x: &mut f32) {
    *x = 2.0;
}

// Public, but not reachable from any launched kernel.
pub fn plain_pub(x: &mut f32) {
    *x = 3.0;
}

// Offload previously kept the normal mono roots, so we'd need to add various `#[cfg(...)]`
// attributes to functions like main that shouldn't end up on the Device. This tests that our new
// mono logic keeps working and correctly disregards this function during device compilation.
fn main() {
    let mut x = 0.0f32;
    core::offload::offload! {
        kernel = launched,
        args = (&mut x,),
    }
}
