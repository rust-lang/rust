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
    name : String
}

trait Barks {
    fn bark(&self) -> String;
}

impl Barks for Dog {
    fn bark(&self) -> String {
        return format!("woof! (I'm {})", self.name).to_strbuf();
    }
}


pub fn main() {
    let snoopy = box Dog{name: "snoopy".to_strbuf()};
    let bubbles = box Dog{name: "bubbles".to_strbuf()};
    let barker = [snoopy as Box<Barks>, bubbles as Box<Barks>];

    for pup in barker.iter() {
        println!("{}", pup.bark());
    }
}

