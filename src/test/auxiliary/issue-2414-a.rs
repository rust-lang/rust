// Copyright 2012 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

#[pkgid="a#0.1"];
// NOTE: remove after the next snapshot
#[link(name = "a", vers = "0.1")];
#[crate_type = "lib"];

type t1 = uint;

trait foo {
    fn foo(&self);
}

impl foo for ~str {
    fn foo(&self) {}
}
