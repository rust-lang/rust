// Copyright 2014 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

use target::Target;

pub fn target() -> Target {
    let mut base = super::android_base::opts();
    base.cpu = "pentium4".to_string();

    Target {
        llvm_target: "i686-linux-android".to_string(),
        target_endian: "little".to_string(),
        target_pointer_width: "32".to_string(),
        arch: "x86".to_string(),
        target_os: "android".to_string(),
        target_env: "gnu".to_string(),
        target_vendor: "unknown".to_string(),
        options: base,
    }
}
