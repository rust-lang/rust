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
        dynamic_linking: true,
        executables: true,
        target_family: Some("unix".to_string()),
        linker_is_gnu: true,
        has_rpath: true,
        pre_link_args: vec![
            // GNU-style linkers will use this to omit linking to libraries
            // which don't actually fulfill any relocations, but only for
            // libraries which follow this flag.  Thus, use it before
            // specifying libraries to link to.
            "-Wl,--as-needed".to_string(),

            // Always enable NX protection when it is available
            "-Wl,-z,noexecstack".to_string(),
        ],
        position_independent_executables: true,
        exe_allocation_crate: super::maybe_jemalloc(),
        .. Default::default()
    }
}
