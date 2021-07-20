//! Implementation of Rust panics via process aborts
//!
//! When compared to the implementation via unwinding, this crate is *much*
//! simpler! That being said, it's not quite as versatile, but here goes!

#![no_std]
#![unstable(feature = "panic_abort", issue = "32837")]
#![doc(issue_tracker_base_url = "https://github.com/rust-lang/rust/issues/")]
#![panic_runtime]
#![allow(unused_features)]
#![feature(core_intrinsics)]
#![feature(nll)]
#![feature(panic_runtime)]
#![feature(std_internals)]
#![feature(staged_api)]
#![feature(rustc_attrs)]
#![feature(asm)]

#[cfg(target_os = "android")]
mod android;

use core::any::Any;
use core::panic::BoxMeUp;

#[rustc_std_internal_symbol]
#[allow(improper_ctypes_definitions)]
pub unsafe extern "C" fn __rust_panic_cleanup(_: *mut u8) -> *mut (dyn Any + Send + 'static) {
    unreachable!()
}

/// "Leak" the payload and abort.
#[rustc_std_internal_symbol]
pub unsafe extern "C" fn __rust_start_panic(_payload: *mut &mut dyn BoxMeUp) -> u32 {
    // Android has the ability to attach a message as part of the abort.
    #[cfg(target_os = "android")]
    android::android_set_abort_message(_payload);

    do_abort();
}

/// Shim to the relevant abort on the platform in question.
fn do_abort() -> ! {
    unsafe {
        abort();
    }

    cfg_if::cfg_if! {
        if #[cfg(unix)] {
            unsafe fn abort() -> ! {
                libc::abort();
            }
        } else if #[cfg(any(target_os = "hermit",
                            all(target_vendor = "fortanix", target_env = "sgx")
        ))] {
            unsafe fn abort() -> ! {
                // call std::sys::abort_internal
                extern "C" {
                    pub fn __rust_abort() -> !;
                }
                __rust_abort();
            }
        } else if #[cfg(all(windows, not(miri)))] {
            // On Windows, use the processor-specific __fastfail mechanism. In Windows 8
            // and later, this will terminate the process immediately without running any
            // in-process exception handlers. In earlier versions of Windows, this
            // sequence of instructions will be treated as an access violation,
            // terminating the process but without necessarily bypassing all exception
            // handlers.
            //
            // https://docs.microsoft.com/en-us/cpp/intrinsics/fastfail
            //
            // Note: this is the same implementation as in libstd's `abort_internal`
            unsafe fn abort() -> ! {
                const FAST_FAIL_FATAL_APP_EXIT: usize = 7;
                cfg_if::cfg_if! {
                    if #[cfg(any(target_arch = "x86", target_arch = "x86_64"))] {
                        asm!("int $$0x29", in("ecx") FAST_FAIL_FATAL_APP_EXIT);
                    } else if #[cfg(all(target_arch = "arm", target_feature = "thumb-mode"))] {
                        asm!(".inst 0xDEFB", in("r0") FAST_FAIL_FATAL_APP_EXIT);
                    } else if #[cfg(target_arch = "aarch64")] {
                        asm!("brk 0xF003", in("x0") FAST_FAIL_FATAL_APP_EXIT);
                    } else {
                        core::intrinsics::abort();
                    }
                }
                core::intrinsics::unreachable();
            }
        } else {
            unsafe fn abort() -> ! {
                core::intrinsics::abort();
            }
        }
    }
}

cfg_if::cfg_if! {
    if #[cfg(target_os = "emscripten")] {
        #[path = "emcc.rs"]
        mod imp;
    } else if #[cfg(target_env = "msvc")] {
        // This is required by the compiler to exist (e.g., it's a lang item), but
        // it's never actually called by the compiler because __C_specific_handler
        // or _except_handler3 is the personality function that is always used.
        // Hence this is just an aborting stub.
        #[rustc_std_internal_symbol]
        fn rust_eh_personality() {
            core::intrinsics::abort()
        }
    } else if #[cfg(any(
        all(target_family = "windows", target_env = "gnu"),
        target_os = "psp",
        target_family = "unix",
        all(target_vendor = "fortanix", target_env = "sgx"),
    ))] {
        #[path = "gcc.rs"]
        mod imp;
    } else {
        // Targets that don't support unwinding.
        // - arch=wasm32
        // - os=none ("bare metal" targets)
        // - os=uefi
        // - os=hermit
        // - nvptx64-nvidia-cuda
        // - arch=avr
    }
}
