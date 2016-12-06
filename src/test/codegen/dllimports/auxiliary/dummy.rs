// Copyright 2016 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

// no-prefer-dynamic
#![crate_type = "staticlib"]

// Since codegen tests don't actually perform linking, this library doesn't need to export
// any symbols.  It's here just to satisfy the compiler looking for a .lib file when processing
// #[link(...)] attributes in wrapper.rs.
