// Copyright 2016 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

#![no_std]
#![unstable(feature = "panic_unwind", issue = "32837")]

#![feature(cfg_target_vendor)]
#![feature(link_cfg)]
#![feature(nll)]
#![feature(staged_api)]
#![feature(unwind_attributes)]
#![feature(static_nobundle)]

#![cfg_attr(not(target_env = "msvc"), feature(libc))]

#[macro_use]
mod macros;

cfg_if! {
    if #[cfg(target_env = "msvc")] {
        // no extra unwinder support needed
    } else if #[cfg(all(target_arch = "wasm32", not(target_os = "emscripten")))] {
        // no unwinder on the system!
    } else {
        extern crate libc;
        mod libunwind;
        pub use libunwind::*;
    }
}

#[cfg(target_env = "musl")]
#[link(name = "unwind", kind = "static", cfg(target_feature = "crt-static"))]
#[link(name = "gcc_s", cfg(not(target_feature = "crt-static")))]
extern {}
