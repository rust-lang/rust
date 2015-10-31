// Copyright 2014 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

use target::TargetOptions;
use std::default::Default;

pub fn opts() -> TargetOptions {
    TargetOptions {
        // OSX has -dead_strip, which doesn't rely on ffunction_sections
        function_sections: false,
        linker: "cc".to_string(),
        dynamic_linking: true,
        executables: true,
        is_like_osx: true,
        has_rpath: true,
        dll_prefix: "lib".to_string(),
        dll_suffix: ".dylib".to_string(),
        archive_format: "bsd".to_string(),
        pre_link_args: Vec::new(),
        exe_allocation_crate: super::maybe_jemalloc(),
        .. Default::default()
    }
}
