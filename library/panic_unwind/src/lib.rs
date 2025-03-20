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
#![feature(cfg_emscripten_wasm_eh)]
#![feature(core_intrinsics)]
#![feature(lang_items)]
#![feature(panic_unwind)]
#![feature(staged_api)]
#![feature(std_internals)]
#![feature(rustc_attrs)]
#![panic_runtime]
#![feature(panic_runtime)]
// `real_imp` is unused with Miri, so silence warnings.
#![cfg_attr(miri, allow(dead_code))]
#![allow(internal_features)]
#![warn(unreachable_pub)]
#![deny(unsafe_op_in_unsafe_fn)]

use alloc::boxed::Box;
use core::any::Any;
use core::panic::PanicPayload;

cfg_if::cfg_if! {
    if #[cfg(all(target_os = "emscripten", not(emscripten_wasm_eh)))] {
        #[path = "emcc.rs"]
        mod imp;
    } else if #[cfg(target_os = "hermit")] {
        #[path = "hermit.rs"]
        mod imp;
    } else if #[cfg(target_os = "l4re")] {
        // L4Re is unix family but does not yet support unwinding.
        #[path = "dummy.rs"]
        mod imp;
    } else if #[cfg(any(
        all(target_family = "windows", target_env = "gnu"),
        target_os = "psp",
        target_os = "xous",
        target_os = "solid_asp3",
        all(target_family = "unix", not(any(target_os = "espidf", target_os = "nuttx"))),
        all(target_vendor = "fortanix", target_env = "sgx"),
        target_family = "wasm",
    ))] {
        #[path = "gcc.rs"]
        mod imp;
    } else if #[cfg(miri)] {
        // Use the Miri runtime on Windows as miri doesn't support funclet based unwinding,
        // only landingpad based unwinding. Also use the Miri runtime on unsupported platforms.
        #[path = "miri.rs"]
        mod imp;
    } else if #[cfg(all(target_env = "msvc", not(target_arch = "arm")))] {
        // LLVM does not support unwinding on 32 bit ARM msvc (thumbv7a-pc-windows-msvc)
        #[path = "seh.rs"]
        mod imp;
    } else {
        // Targets that don't support unwinding.
        // - os=none ("bare metal" targets)
        // - os=uefi
        // - os=espidf
        // - nvptx64-nvidia-cuda
        // - arch=avr
        #[path = "dummy.rs"]
        mod imp;
    }
}

unsafe extern "C" {
    /// Handler in std called when a panic object is dropped outside of
    /// `catch_unwind`.
    #[cfg_attr(not(bootstrap), rustc_std_internal_symbol)]
    fn __rust_drop_panic() -> !;

    /// Handler in std called when a foreign exception is caught.
    #[cfg_attr(not(bootstrap), rustc_std_internal_symbol)]
    fn __rust_foreign_exception() -> !;
}

#[rustc_std_internal_symbol]
#[allow(improper_ctypes_definitions)]
pub unsafe extern "C" fn __rust_panic_cleanup(payload: *mut u8) -> *mut (dyn Any + Send + 'static) {
    unsafe { Box::into_raw(imp::cleanup(payload)) }
}

// Entry point for raising an exception, just delegates to the platform-specific
// implementation.
#[rustc_std_internal_symbol]
pub unsafe fn __rust_start_panic(payload: &mut dyn PanicPayload) -> u32 {
    unsafe {
        let payload = Box::from_raw(payload.take_box());

        imp::panic(payload)
    }
}
