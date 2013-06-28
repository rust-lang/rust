// Copyright 2013 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

// Context data structure used by rustpkg


use std::hashmap::HashMap;

pub struct Ctx {
    // Sysroot -- if this is None, uses rustc filesearch's
    // idea of the default
    sysroot_opt: Option<@Path>,
    // I'm not sure what this is for
    json: bool,
    // Cache of hashes of things already installed
    // though I'm not sure why the value is a bool
    dep_cache: @mut HashMap<~str, bool>,
}
