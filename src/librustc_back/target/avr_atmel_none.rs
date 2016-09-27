// Copyright 2015 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

use target::{Target, TargetResult};

pub fn target() -> TargetResult {
    Ok(Target {
        llvm_target: "avr-atmel-none".to_string(),
        target_endian: "little".to_string(),
        target_pointer_width: "16".to_string(),
        data_layout: "e-p:16:16:16-i8:8:8-i16:16:16-i32:32:32-i64:64:64-f32:32:32-f64:64:64-n8".to_string(),
        arch: "avr".to_string(),
        target_os: "none".to_string(),
        target_env: "gnu".to_string(),
        target_vendor: "unknown".to_string(),
        options: super::none_base::opts()
    })
}
