// Copyright 2017 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

// produce special borrowck message inside all kinds of loops

struct FuncWrapper<'a, T : 'a> {
    func : fn(&'a mut T) -> ()
}

impl<'a, T : 'a> FuncWrapper<'a, T> {
    fn in_loop(self, arg : &'a mut T) {
        loop {
            (self.func)(arg) //~ ERROR cannot borrow
        }
    }

    fn in_while(self, arg : &'a mut T) {
        while true {
            (self.func)(arg) //~ ERROR cannot borrow
        }
    }

    fn in_for(self, arg : &'a mut T) {
        let v : Vec<()> = vec![];
        for _ in v.iter() {
            (self.func)(arg) //~ ERROR cannot borrow
        }
    }
}

fn main() {
}

