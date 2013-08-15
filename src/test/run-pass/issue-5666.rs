// Copyright 2013 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

struct Dog {
    name : ~str
}

trait Barks {
    fn bark(&self) -> ~str;
}

impl Barks for Dog {
    fn bark(&self) -> ~str {
        return fmt!("woof! (I'm %s)", self.name);
    }
}


pub fn main() {
    let snoopy = ~Dog{name: ~"snoopy"};
    let bubbles = ~Dog{name: ~"bubbles"};
    let barker = [snoopy as ~Barks, bubbles as ~Barks];

    for pup in barker.iter() {
        println(fmt!("%s", pup.bark()));
    }
}

