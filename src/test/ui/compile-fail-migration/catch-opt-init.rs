// Copyright 2017 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

#![feature(catch_expr)]

fn use_val<T: Sized>(_x: T) {}

pub fn main() {
    let cfg_res;
    let _: Result<(), ()> = do catch {
        Err(())?;
        cfg_res = 5;
        Ok::<(), ()>(())?;
        use_val(cfg_res);
    };
    assert_eq!(cfg_res, 5); //~ ERROR use of possibly uninitialized variable
}

