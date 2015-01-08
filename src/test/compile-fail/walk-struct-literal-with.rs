// Copyright 2014 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

struct Mine{
    test: String,
    other_val: isize
}

impl Mine{
    fn make_string_bar(mut self) -> Mine{
        self.test = "Bar".to_string();
        self
    }
}

fn main(){
    let start = Mine{test:"Foo".to_string(), other_val:0};
    let end = Mine{other_val:1, ..start.make_string_bar()};
    println!("{}", start.test); //~ ERROR use of moved value: `start.test`
}

