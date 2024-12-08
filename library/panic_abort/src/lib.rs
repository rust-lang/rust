//! Implementation of Rust panics via process aborts
//!
//! When compared to the implementation via unwinding, this crate is *much*
//! simpler! That being said, it's not quite as versatile, but here goes!

#![no_std]
#![unstable(feature = "panic_abort", issue = "32837")]
#![doc(issue_tracker_base_url = "https://github.com/rust-lang/rust/issues/")]
#![panic_runtime]
#![allow(unused_features)]
#![feature(asm_experimental_arch)]
#![feature(core_intrinsics)]
#![feature(panic_runtime)]
#![feature(std_internals)]
#![feature(staged_api)]
#![feature(rustc_attrs)]
#![allow(internal_features)]

#[cfg(target_os = "android")]
mod android;

#[cfg(target_os = "zkvm")]
mod zkvm;

use core::any::Any;
use core::panic::PanicPayload;

#[rustc_std_internal_symbol]
#[allow(improper_ctypes_definitions)]
pub unsafe extern "C" fn __rust_panic_cleanup(_: *mut u8) -> *mut (dyn Any + Send + 'static) {
    unreachable!()
}

// "Leak" the payload and shim to the relevant abort on the platform in question.
#[rustc_std_internal_symbol]
pub unsafe fn __rust_start_panic(_payload: &mut dyn PanicPayload) -> u32 {
    // Android has the ability to attach a message as part of the abort.
    #[cfg(target_os = "android")]
    android::android_set_abort_message(_payload);
    #[cfg(target_os = "zkvm")]
    zkvm::zkvm_set_abort_message(_payload);

    abort();

    cfg_if::cfg_if! {
        if #[cfg(any(unix, target_os = "solid_asp3"))] {
            unsafe fn abort() -> ! {
                libc::abort();
            }
        } else if #[cfg(any(target_os = "hermit",
                            all(target_vendor = "fortanix", target_env = "sgx"),
                            target_os = "xous",
                            target_os = "uefi",
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
            // Note: this is the same implementation as in std's `abort_internal`
            unsafe fn abort() -> ! {
                #[allow(unused)]
                const FAST_FAIL_FATAL_APP_EXIT: usize = 7;
                cfg_if::cfg_if! {
                    if #[cfg(any(target_arch = "x86", target_arch = "x86_64"))] {
                        core::arch::asm!("int $$0x29", in("ecx") FAST_FAIL_FATAL_APP_EXIT, options(noreturn, nostack));
                    } else if #[cfg(all(target_arch = "arm", target_feature = "thumb-mode"))] {
                        core::arch::asm!(".inst 0xDEFB", in("r0") FAST_FAIL_FATAL_APP_EXIT, options(noreturn, nostack));
                    } else if #[cfg(any(target_arch = "aarch64", target_arch = "arm64ec"))] {
                        core::arch::asm!("brk 0xF003", in("x0") FAST_FAIL_FATAL_APP_EXIT, options(noreturn, nostack));
                    } else {
                        core::intrinsics::abort();
                    }
                }
            }
        } else if #[cfg(target_os = "teeos")] {
            mod teeos {
                extern "C" {
                    pub fn TEE_Panic(code: u32) -> !;
                }
            }

            unsafe fn abort() -> ! {
                teeos::TEE_Panic(1);
            }
        } else {
            unsafe fn abort() -> ! {
                core::intrinsics::abort();
            }
        }
    }
}

// This... is a bit of an oddity. The tl;dr; is that this is required to link
// correctly, the longer explanation is below.
//
// Right now the binaries of core/std that we ship are all compiled with
// `-C panic=unwind`. This is done to ensure that the binaries are maximally
// compatible with as many situations as possible. The compiler, however,
// requires a "personality function" for all functions compiled with `-C
// panic=unwind`. This personality function is hardcoded to the symbol
// `rust_eh_personality` and is defined by the `eh_personality` lang item.
//
// So... why not just define that lang item here? Good question! The way that
// panic runtimes are linked in is actually a little subtle in that they're
// "sort of" in the compiler's crate store, but only actually linked if another
// isn't actually linked. This ends up meaning that both this crate and the
// panic_unwind crate can appear in the compiler's crate store, and if both
// define the `eh_personality` lang item then that'll hit an error.
//
// To handle this the compiler only requires the `eh_personality` is defined if
// the panic runtime being linked in is the unwinding runtime, and otherwise
// it's not required to be defined (rightfully so). In this case, however, this
// library just defines this symbol so there's at least some personality
// somewhere.
//
// Essentially this symbol is just defined to get wired up to core/std
// binaries, but it should never be called as we don't link in an unwinding
// runtime at all.
pub mod personalities {
    // In the past this module used to contain stubs for the personality
    // functions of various platforms, but these where removed when personality
    // functions were moved to std.

    // This corresponds to the `eh_catch_typeinfo` lang item
    // that's only used on Emscripten currently.
    //
    // Since panics don't generate exceptions and foreign exceptions are
    // currently UB with -C panic=abort (although this may be subject to
    // change), any catch_unwind calls will never use this typeinfo.
    #[rustc_std_internal_symbol]
    #[allow(non_upper_case_globals)]
    #[cfg(target_os = "emscripten")]
    static rust_eh_catch_typeinfo: [usize; 2] = [0; 2];
}
