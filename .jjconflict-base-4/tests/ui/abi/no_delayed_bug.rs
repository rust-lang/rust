// Used to ICE due to the ABI checker emitting a delayed bug when failing to get
// the FnAbi due to a const assert, while codegen skipped the call due to being
// unreachable.
//@ compile-flags: -Copt-level=0
//@ build-fail

//~? ERROR the SIMD type `Simd<u8, 256>` has more elements than the limit 64

#![feature(portable_simd)]

fn main() {
    if false {
        let _ = core::simd::Simd::<u8, 256>::splat(0);
    }
}
