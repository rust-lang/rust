// Copyright 2016 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

#![cfg_attr(not(stage0), feature(compiler_builtins))]
#![no_std]
#![cfg_attr(not(stage0), compiler_builtins)]
#![unstable(feature = "compiler_builtins_lib",
            reason = "internal implementation detail of rustc right now",
            issue = "0")]
#![crate_name = "compiler_builtins"]
#![crate_type = "rlib"]
#![feature(staged_api)]
