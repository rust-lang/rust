//! Implementation of Rust panics via process aborts
//!
//! When compared to the implementation via unwinding, this crate is *much*
//! simpler! That being said, it's not quite as versatile, but here goes!

#![no_std]
#![unstable(feature = "panic_abort", issue = "32837")]
#![doc(issue_tracker_base_url = "https://github.com/rust-lang/rust/issues/")]
#![panic_runtime]
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
    unsafe {
        android::android_set_abort_message(_payload);
    }
    #[cfg(target_os = "zkvm")]
    unsafe {
        zkvm::zkvm_set_abort_message(_payload);
    }

    unsafe extern "Rust" {
        // This is defined in std::rt.
        #[rustc_std_internal_symbol]
        safe fn __rust_abort() -> !;
    }

    __rust_abort()
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
