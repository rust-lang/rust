// Copyright 2017 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

use std::collections::HashMap;

fn main() {
    let dict: HashMap<i32, i32> = HashMap::new();
    let debug_dump_dict = || {
        for (key, value) in dict {
            println!("{:?} - {:?}", key, value);
        }
    };
    debug_dump_dict();
    debug_dump_dict();
    //~^ ERROR use of moved value: `debug_dump_dict`
}
