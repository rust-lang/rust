// Copyright 2016 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

//! Implementation of panics via stack unwinding
//!
//! This crate is an implementation of panics in Rust using "most native" stack
//! unwinding mechanism of the platform this is being compiled for. This
//! essentially gets categorized into three buckets currently:
//!
//! 1. MSVC targets use SEH in the `seh.rs` file.
//! 2. The 64-bit MinGW target half-uses SEH and half-use gcc-like information
//!    in the `seh64_gnu.rs` module.
//! 3. All other targets use libunwind/libgcc in the `gcc/mod.rs` module.
//!
//! More documentation about each implementation can be found in the respective
//! module.

#![no_std]
#![crate_name = "panic_unwind"]
#![crate_type = "rlib"]
#![unstable(feature = "panic_unwind", issue = "32837")]
#![doc(html_logo_url = "https://www.rust-lang.org/logos/rust-logo-128x128-blk-v2.png",
       html_favicon_url = "https://doc.rust-lang.org/favicon.ico",
       html_root_url = "https://doc.rust-lang.org/nightly/",
       issue_tracker_base_url = "https://github.com/rust-lang/rust/issues/")]
#![deny(warnings)]

#![feature(alloc)]
#![feature(core_intrinsics)]
#![feature(lang_items)]
#![feature(libc)]
#![feature(panic_unwind)]
#![feature(raw)]
#![feature(staged_api)]
#![feature(unwind_attributes)]
#![cfg_attr(target_env = "msvc", feature(raw))]

#![panic_runtime]
#![feature(panic_runtime)]

extern crate alloc;
extern crate libc;
extern crate unwind;

use core::intrinsics;
use core::mem;
use core::raw;

// Rust runtime's startup objects depend on these symbols, so make them public.
#[cfg(all(target_os="windows", target_arch = "x86", target_env="gnu"))]
pub use imp::eh_frame_registry::*;

// *-pc-windows-msvc
#[cfg(target_env = "msvc")]
#[path = "seh.rs"]
mod imp;

// x86_64-pc-windows-gnu
#[cfg(all(windows, target_arch = "x86_64", target_env = "gnu"))]
#[path = "seh64_gnu.rs"]
mod imp;

// i686-pc-windows-gnu and all others
#[cfg(any(all(unix, not(target_os = "emscripten")),
          target_os = "redox",
          all(windows, target_arch = "x86", target_env = "gnu")))]
#[path = "gcc.rs"]
mod imp;

// emscripten
#[cfg(target_os = "emscripten")]
#[path = "emcc.rs"]
mod imp;

mod dwarf;
mod windows;

// Entry point for catching an exception, implemented using the `try` intrinsic
// in the compiler.
//
// The interaction between the `payload` function and the compiler is pretty
// hairy and tightly coupled, for more information see the compiler's
// implementation of this.
#[no_mangle]
pub unsafe extern "C" fn __rust_maybe_catch_panic(f: fn(*mut u8),
                                                  data: *mut u8,
                                                  data_ptr: *mut usize,
                                                  vtable_ptr: *mut usize)
                                                  -> u32 {
    let mut payload = imp::payload();
    if intrinsics::try(f, data, &mut payload as *mut _ as *mut _) == 0 {
        0
    } else {
        let obj = mem::transmute::<_, raw::TraitObject>(imp::cleanup(payload));
        *data_ptr = obj.data as usize;
        *vtable_ptr = obj.vtable as usize;
        1
    }
}

// Entry point for raising an exception, just delegates to the platform-specific
// implementation.
#[no_mangle]
#[unwind]
pub unsafe extern "C" fn __rust_start_panic(data: usize, vtable: usize) -> u32 {
    imp::panic(mem::transmute(raw::TraitObject {
        data: data as *mut (),
        vtable: vtable as *mut (),
    }))
}
