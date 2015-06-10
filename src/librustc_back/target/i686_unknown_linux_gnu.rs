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
    let mut base = super::linux_base::opts();
    base.cpu = "pentium4".to_owned();
    base.pre_link_args.push("-m32".to_owned());

    Target {
        data_layout: "e-p:32:32-f64:32:64-i64:32:64-f80:32:32-n8:16:32".to_owned(),
        llvm_target: "i686-unknown-linux-gnu".to_owned(),
        target_endian: "little".to_owned(),
        target_pointer_width: "32".to_owned(),
        arch: "x86".to_owned(),
        target_os: "linux".to_owned(),
        target_env: "gnu".to_owned(),
        options: base,
    }
}
