// Copyright 2014 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

// revisions:rpass1 rpass2
// compile-flags: -Z query-dep-graph

#![rustc_partition_reused(module="__rustc_fallback_codegen_unit", cfg="rpass2")]
#![feature(rustc_attrs)]

#![crate_type="rlib"]
pub fn foo<T>() { }
