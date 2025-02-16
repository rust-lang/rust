//! Ensure we trigger abi_unsupported_vector_types for target features that are usually enabled
//! on a target, but disabled in this file via a `-C` flag.
//@ compile-flags: --crate-type=rlib --target=i586-unknown-linux-gnu -C target-feature=-sse,-sse2
//@ build-pass
//@ ignore-pass (test emits codegen-time warnings)
//@ needs-llvm-components: x86
#![feature(no_core, lang_items, repr_simd)]
#![no_core]
#![allow(improper_ctypes_definitions)]

#[lang = "sized"]
trait Sized {}

#[lang = "copy"]
trait Copy {}

#[repr(simd)]
pub struct SseVector([i64; 2]);

#[no_mangle]
pub unsafe extern "C" fn f(_: SseVector) {
    //~^ this function definition uses SIMD vector type `SseVector` which (with the chosen ABI) requires the `sse` target feature, which is not enabled
    //~| WARNING this was previously accepted by the compiler
}
