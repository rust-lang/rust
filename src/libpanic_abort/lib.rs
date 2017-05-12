// Copyright 2016 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

//! Implementation of Rust panics via process aborts
//!
//! When compared to the implementation via unwinding, this crate is *much*
//! simpler! That being said, it's not quite as versatile, but here goes!

#![no_std]
#![crate_name = "panic_abort"]
#![crate_type = "rlib"]
#![unstable(feature = "panic_abort", issue = "32837")]
#![doc(html_logo_url = "https://www.rust-lang.org/logos/rust-logo-128x128-blk-v2.png",
       html_favicon_url = "https://doc.rust-lang.org/favicon.ico",
       html_root_url = "https://doc.rust-lang.org/nightly/",
       issue_tracker_base_url = "https://github.com/rust-lang/rust/issues/")]
#![deny(warnings)]

#![feature(staged_api)]

#![panic_runtime]
#![feature(panic_runtime)]
#![cfg_attr(unix, feature(libc))]
#![cfg_attr(any(target_os = "redox", windows), feature(core_intrinsics))]

// Rust's "try" function, but if we're aborting on panics we just call the
// function as there's nothing else we need to do here.
#[no_mangle]
pub unsafe extern fn __rust_maybe_catch_panic(f: fn(*mut u8),
                                              data: *mut u8,
                                              _data_ptr: *mut usize,
                                              _vtable_ptr: *mut usize) -> u32 {
    f(data);
    0
}

// "Leak" the payload and shim to the relevant abort on the platform in
// question.
//
// For Unix we just use `abort` from libc as it'll trigger debuggers, core
// dumps, etc, as one might expect. On Windows, however, the best option we have
// is the `__fastfail` intrinsics, but that's unfortunately not defined in LLVM,
// and the `RaiseFailFastException` function isn't available until Windows 7
// which would break compat with XP. For now just use `intrinsics::abort` which
// will kill us with an illegal instruction, which will do a good enough job for
// now hopefully.
#[no_mangle]
pub unsafe extern fn __rust_start_panic(_data: usize, _vtable: usize) -> u32 {
    abort();

    #[cfg(unix)]
    unsafe fn abort() -> ! {
        extern crate libc;
        libc::abort();
    }

    #[cfg(any(target_os = "redox", windows))]
    unsafe fn abort() -> ! {
        core::intrinsics::abort();
    }
}

// This... is a bit of an oddity. The tl;dr; is that this is required to link
// correctly, the longer explanation is below.
//
// Right now the binaries of libcore/libstd that we ship are all compiled with
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
// Essentially this symbol is just defined to get wired up to libcore/libstd
// binaries, but it should never be called as we don't link in an unwinding
// runtime at all.
pub mod personalities {

    #[no_mangle]
    #[cfg(not(all(target_os = "windows",
                  target_env = "gnu",
                  target_arch = "x86_64")))]
    pub extern fn rust_eh_personality() {}

    // On x86_64-pc-windows-gnu we use our own personality function that needs
    // to return `ExceptionContinueSearch` as we're passing on all our frames.
    #[no_mangle]
    #[cfg(all(target_os = "windows",
              target_env = "gnu",
              target_arch = "x86_64"))]
    pub extern fn rust_eh_personality(_record: usize,
                                      _frame: usize,
                                      _context: usize,
                                      _dispatcher: usize) -> u32 {
        1 // `ExceptionContinueSearch`
    }

    // Similar to above, this corresponds to the `eh_unwind_resume` lang item
    // that's only used on Windows currently.
    //
    // Note that we don't execute landing pads, so this is never called, so it's
    // body is empty.
    #[no_mangle]
    #[cfg(all(target_os = "windows", target_env = "gnu"))]
    pub extern fn rust_eh_unwind_resume() {}

    // These two are called by our startup objects on i686-pc-windows-gnu, but
    // they don't need to do anything so the bodies are nops.
    #[no_mangle]
    #[cfg(all(target_os = "windows", target_env = "gnu", target_arch = "x86"))]
    pub extern fn rust_eh_register_frames() {}
    #[no_mangle]
    #[cfg(all(target_os = "windows", target_env = "gnu", target_arch = "x86"))]
    pub extern fn rust_eh_unregister_frames() {}
}
