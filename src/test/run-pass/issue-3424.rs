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
extern mod std;
use core::path::{Path};

type rsrc_loader = ~fn(path: &Path) -> result::Result<~str, ~str>;

#[test]
fn tester()
{
    let loader: rsrc_loader = |_path| {result::Ok(~"more blah")};

    let path = path::from_str("blah");
    assert!(loader(&path).is_ok());
}

pub fn main() {}
