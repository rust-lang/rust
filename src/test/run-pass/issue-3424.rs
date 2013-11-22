// xfail-fast

// Copyright 2012 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

// rustc --test ignores2.rs && ./ignores2
extern mod extra;

use std::path::{Path};
use std::path;
use std::result;

type rsrc_loader = proc(path: &Path) -> result::Result<~str, ~str>;

fn tester()
{
    let loader: rsrc_loader = proc(_path) {result::Ok(~"more blah")};

    let path = path::Path::init("blah");
    assert!(loader(&path).is_ok());
}

pub fn main() {}
