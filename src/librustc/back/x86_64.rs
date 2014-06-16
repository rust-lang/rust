// Copyright 2012 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.


use back::target_strs;
use syntax::abi;

pub fn get_target_strs(target_triple: String, target_os: abi::Os) -> target_strs::t {
    return target_strs::t {
        module_asm: "".to_string(),

        data_layout: match target_os {
          abi::OsMacos => {
            "e-p:64:64:64-i1:8:8-i8:8:8-i16:16:16-i32:32:32-i64:64:64-\
                f32:32:32-f64:64:64-v64:64:64-v128:128:128-a0:0:64-\
                s0:64:64-f80:128:128-n8:16:32:64".to_string()
          }

          abi::OsiOS => {
            "e-p:64:64:64-i1:8:8-i8:8:8-i16:16:16-i32:32:32-i64:64:64-\
                f32:32:32-f64:64:64-v64:64:64-v128:128:128-a0:0:64-\
                s0:64:64-f80:128:128-n8:16:32:64".to_string()
          }

          abi::OsWin32 => {
            // FIXME: Test this. Copied from linux (#2398)
            "e-p:64:64:64-i1:8:8-i8:8:8-i16:16:16-i32:32:32-i64:64:64-\
                f32:32:32-f64:64:64-v64:64:64-v128:128:128-a0:0:64-\
                s0:64:64-f80:128:128-n8:16:32:64-S128".to_string()
          }

          abi::OsLinux => {
            "e-p:64:64:64-i1:8:8-i8:8:8-i16:16:16-i32:32:32-i64:64:64-\
                f32:32:32-f64:64:64-v64:64:64-v128:128:128-a0:0:64-\
                s0:64:64-f80:128:128-n8:16:32:64-S128".to_string()
          }
          abi::OsAndroid => {
            "e-p:64:64:64-i1:8:8-i8:8:8-i16:16:16-i32:32:32-i64:64:64-\
                f32:32:32-f64:64:64-v64:64:64-v128:128:128-a0:0:64-\
                s0:64:64-f80:128:128-n8:16:32:64-S128".to_string()
          }

          abi::OsFreebsd => {
            "e-p:64:64:64-i1:8:8-i8:8:8-i16:16:16-i32:32:32-i64:64:64-\
                f32:32:32-f64:64:64-v64:64:64-v128:128:128-a0:0:64-\
                s0:64:64-f80:128:128-n8:16:32:64-S128".to_string()
          }
        },

        target_triple: target_triple,

        cc_args: vec!("-m64".to_string()),
    };
}
