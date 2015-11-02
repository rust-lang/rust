// Copyright 2014 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

#![allow(missing_docs)]

#[macro_use]
pub mod rt;

#[cfg(any(target_family = "unix", target_family = "windows"))] pub mod thread_local;
#[cfg(any(target_family = "unix", target_family = "windows"))] pub mod net_bsd;
#[cfg(any(target_family = "unix", target_family = "windows"))] pub mod unwind;
#[cfg(any(target_family = "unix", target_family = "windows"))] pub mod libunwind;
#[cfg(target_family = "windows")] pub mod dwarf;

#[cfg(any(all(unix, not(any(target_os = "macos", target_os = "ios"))),
          all(windows, target_env = "gnu")))]
pub mod gnu;

pub mod stdio;
pub mod env;
pub mod net;
pub mod process;
pub mod os_str;
pub mod error;
pub mod c;
