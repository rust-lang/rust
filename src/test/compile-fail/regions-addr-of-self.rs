// Copyright 2012 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

struct dog {
    cats_chased: uint,
}

impl dog {
    pub fn chase_cat(&mut self) {
        let p: &'static mut uint = &mut self.cats_chased; //~ ERROR cannot infer
        *p += 1u;
    }

    pub fn chase_cat_2(&mut self) {
        let p: &mut uint = &mut self.cats_chased;
        *p += 1u;
    }
}

fn dog() -> dog {
    dog {
        cats_chased: 0u
    }
}

fn main() {
    let mut d = dog();
    d.chase_cat();
    println!("cats_chased: {}", d.cats_chased);
}
