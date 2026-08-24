//! Implementation of panics via stack unwinding
//!
//! This crate is an implementation of panics in Rust using "most native" stack
//! unwinding mechanism of the platform this is being compiled for. This
//! essentially gets categorized into four buckets currently:
//!
//! 1. When running inside miri, MSVC targets use Miri intrinsics in the `miri.rs` file.
//! 2. MSVC targets use SEH in the `seh.rs` file.
//! 3. Some targets use an aborting implementation in the `dummy.rs` or `hermit.rs` files.
//! 4. All other targets use libunwind/libgcc in the `gcc.rs` file.
//!
//! More documentation about each implementation can be found in the respective
//! module.

#![no_std]
#![unstable(feature = "panic_unwind", issue = "32837")]
#![doc(issue_tracker_base_url = "https://github.com/rust-lang/rust/issues/")]
#![feature(core_intrinsics)]
#![feature(panic_unwind)]
#![feature(staged_api)]
#![feature(std_internals)]
#![feature(rustc_attrs)]
#![panic_runtime]
#![feature(panic_runtime)]
#![allow(internal_features)]
#![allow(unused_features)]
#![warn(unreachable_pub)]
#![deny(unsafe_op_in_unsafe_fn)]

use alloc::boxed::Box;
use alloc::panicking::PanicPayload;
use core::any::Any;

cfg_select! {
    any(
        all(target_family = "windows", target_env = "gnu"),
        target_os = "psp",
        target_os = "xous",
        target_os = "solid_asp3",
        all(target_family = "unix", not(any(target_os = "espidf", target_os = "nuttx"))),
        all(target_vendor = "fortanix", target_env = "sgx"),
        target_family = "wasm",
    ) => {
        #[path = "gcc.rs"]
        mod imp;
    }
    miri => {
        // Use the Miri runtime on Windows as miri doesn't support funclet based unwinding,
        // only landingpad based unwinding. Also use the Miri runtime on unsupported platforms.
        #[path = "miri.rs"]
        mod imp;
    }
    all(target_env = "msvc", not(target_arch = "arm")) => {
        // LLVM does not support unwinding on 32 bit ARM msvc (thumbv7a-pc-windows-msvc)
        #[path = "seh.rs"]
        mod imp;
    }
    _ => {
        // Targets that don't support unwinding.
        // - os=none ("bare metal" targets)
        // - os=uefi
        // - os=espidf
        // - os=hermit
        // - nvptx64-nvidia-cuda
        // - arch=avr
        #[path = "dummy.rs"]
        mod imp;
    }
}

unsafe extern "Rust" {
    /// Handler in std called when a panic object is dropped outside of
    /// `catch_unwind`.
    #[rustc_std_internal_symbol]
    safe fn __rust_drop_panic() -> !;

    /// Handler in std called when a foreign exception is caught.
    #[rustc_std_internal_symbol]
    safe fn __rust_foreign_exception() -> !;
}

#[rustc_std_internal_symbol]
pub unsafe fn __rust_panic_cleanup(payload: *mut u8) -> Box<dyn Any + Send + 'static> {
    unsafe { imp::cleanup(payload) }
}

// Entry point for raising an exception, just delegates to the platform-specific
// implementation.
#[rustc_std_internal_symbol]
pub fn __rust_start_panic(payload: &mut dyn PanicPayload) -> u32 {
    imp::panic(payload)
}
