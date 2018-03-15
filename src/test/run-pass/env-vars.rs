// Copyright 2015 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.


use std::env::*;

fn main() {
    for (k, v) in vars_os() {
        let v2 = var_os(&k);
        assert!(v2.as_ref().map(|s| &**s) == Some(&*v),
                "bad vars->var transition: {:?} {:?} {:?}", k, v, v2);
    }
}
