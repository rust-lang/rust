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
    Target {
        data_layout: "e-p:32:32-f64:32:64-i64:32:64-f80:32:32-n8:16:32".into_string(),
        llvm_target: "x86_64-unknown-dragonfly".into_string(),
        target_endian: "little".into_string(),
        target_word_size: "32".into_string(),
        arch: "x86_64".into_string(),
        target_os: "dragonfly".into_string(),
        options: super::dragonfly_base::opts()
    }
}
