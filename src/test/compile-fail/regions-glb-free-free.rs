// Copyright 2012 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

mod argparse {
    pub struct Flag<'a> {
        name: &'a str,
        desc: &'a str,
        max_count: uint,
        value: uint
    }

    pub fn flag<'r>(name: &'r str, desc: &'r str) -> Flag<'r> {
        Flag { name: name, desc: desc, max_count: 1, value: 0 }
    }

    impl<'a> Flag<'a> {
        pub fn set_desc(self, s: &str) -> Flag<'a> {
            Flag { //~ ERROR cannot infer
                name: self.name,
                desc: s,
                max_count: self.max_count,
                value: self.value
            }
        }
    }
}

fn main () {
    let f : argparse::Flag = argparse::flag("flag", "My flag");
    let updated_flag = f.set_desc("My new flag");
    assert_eq!(updated_flag.desc.as_slice(), "My new flag");
}
