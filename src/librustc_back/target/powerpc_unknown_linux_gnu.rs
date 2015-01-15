// Copyright 2012 The Rust Project Developers. See the COPYRIGHT
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
    let mut base = super::linux_base::opts();
    base.pre_link_args.push("-m32".to_string());

    Target {
        data_layout: "E-S8-p:32:32-f64:32:64-i64:32:64-f80:32:32-n8:16:32".to_string(),
        llvm_target: "powerpc-unknown-linux-gnu".to_string(),
        target_endian: "big".to_string(),
        target_pointer_width: "32".to_string(),
        arch: "powerpc".to_string(),
        target_os: "linux".to_string(),
        options: base,
    }
}
