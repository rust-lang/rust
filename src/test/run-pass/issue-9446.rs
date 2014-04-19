// Copyright 2013 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

struct Wrapper(~str);

impl Wrapper {
    pub fn new(wrapped: ~str) -> Wrapper {
        Wrapper(wrapped)
    }

    pub fn say_hi(&self) {
        let Wrapper(ref s) = *self;
        println!("hello {}", *s);
    }
}

impl Drop for Wrapper {
    fn drop(&mut self) {}
}

pub fn main() {
    {
        // This runs without complaint.
        let x = Wrapper::new("Bob".to_owned());
        x.say_hi();
    }
    {
        // This fails to compile, circa 0.8-89-gc635fba.
        // error: internal compiler error: drop_ty_immediate: non-box ty
        Wrapper::new("Bob".to_owned()).say_hi();
    }
}
