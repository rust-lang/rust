//! Implementation of panics via stack unwinding
//!
//! This crate is an implementation of panics in Rust using "most native" stack
//! unwinding mechanism of the platform this is being compiled for. This
//! essentially gets categorized into three buckets currently:
//!
//! 1. MSVC targets use SEH in the `seh.rs` file.
//! 2. Emscripten uses C++ exceptions in the `emcc.rs` file.
//! 3. All other targets use libunwind/libgcc in the `gcc.rs` file.
//!
//! More documentation about each implementation can be found in the respective
//! module.

#![no_std]
#![unstable(feature = "panic_unwind", issue = "32837")]
#![doc(issue_tracker_base_url = "https://github.com/rust-lang/rust/issues/")]
#![feature(core_intrinsics)]
#![feature(lang_items)]
#![feature(nll)]
#![feature(panic_unwind)]
#![feature(staged_api)]
#![feature(std_internals)]
#![feature(unwind_attributes)]
#![feature(abi_thiscall)]
#![feature(rustc_attrs)]
#![panic_runtime]
#![feature(panic_runtime)]
// `real_imp` is unused with Miri, so silence warnings.
#![cfg_attr(miri, allow(dead_code))]

use alloc::boxed::Box;
use core::any::Any;
use core::panic::BoxMeUp;

cfg_if::cfg_if! {
    if #[cfg(target_os = "emscripten")] {
        #[path = "emcc.rs"]
        mod real_imp;
    } else if #[cfg(target_os = "hermit")] {
        #[path = "hermit.rs"]
        mod real_imp;
    } else if #[cfg(target_env = "msvc")] {
        #[path = "seh.rs"]
        mod real_imp;
    } else if #[cfg(any(
        all(target_family = "windows", target_env = "gnu"),
        target_os = "psp",
        target_family = "unix",
        all(target_vendor = "fortanix", target_env = "sgx"),
    ))] {
        // Rust runtime's startup objects depend on these symbols, so make them public.
        #[cfg(all(target_os="windows", target_arch = "x86", target_env="gnu"))]
        pub use real_imp::eh_frame_registry::*;
        #[path = "gcc.rs"]
        mod real_imp;
    } else {
        // Targets that don't support unwinding.
        // - arch=wasm32
        // - os=none ("bare metal" targets)
        // - os=uefi
        // - nvptx64-nvidia-cuda
        // - arch=avr
        #[path = "dummy.rs"]
        mod real_imp;
    }
}

cfg_if::cfg_if! {
    if #[cfg(miri)] {
        // Use the Miri runtime.
        // We still need to also load the normal runtime above, as rustc expects certain lang
        // items from there to be defined.
        #[path = "miri.rs"]
        mod imp;
    } else {
        // Use the real runtime.
        use real_imp as imp;
    }
}

extern "C" {
    /// Handler in libstd called when a panic object is dropped outside of
    /// `catch_unwind`.
    fn __rust_drop_panic() -> !;

    /// Handler in libstd called when a foreign exception is caught.
    fn __rust_foreign_exception() -> !;
}

mod dwarf;

#[rustc_std_internal_symbol]
#[allow(improper_ctypes_definitions)]
pub unsafe extern "C" fn __rust_panic_cleanup(payload: *mut u8) -> *mut (dyn Any + Send + 'static) {
    Box::into_raw(imp::cleanup(payload))
}

// Entry point for raising an exception, just delegates to the platform-specific
// implementation.
#[rustc_std_internal_symbol]
#[unwind(allowed)]
pub unsafe extern "C" fn __rust_start_panic(payload: *mut &mut dyn BoxMeUp) -> u32 {
    let payload = Box::from_raw((*payload).take_box());

    imp::panic(payload)
}
