// Copyright 2012-2016 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

#![feature(cfg_target_feature)]

use std::env;

fn main() {
    match env::var("TARGET") {
        Ok(s) => {
            // Skip this tests on i586-unknown-linux-gnu where sse2 is disabled
            if s.contains("i586") {
                return
            }
        }
        Err(_) => return,
    }
    if cfg!(any(target_arch = "x86", target_arch = "x86_64")) {
        assert!(cfg!(target_feature = "sse2"),
                "SSE2 was not detected as available on an x86 platform");
    }
}
