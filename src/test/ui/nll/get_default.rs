// Copyright 2016 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

// Basic test for free regions in the NLL code. This test ought to
// report an error due to a reborrowing constraint. Right now, we get
// a variety of errors from the older, AST-based machinery (notably
// borrowck), and then we get the NLL error at the end.

// compile-flags:-Znll -Zborrowck=compare -Znll-dump-cause

struct Map {
}

impl Map {
    fn get(&self) -> Option<&String> { None }
    fn set(&mut self, v: String) { }
}

fn ok(map: &mut Map) -> &String {
    loop {
        match map.get() {
            Some(v) => {
                return v;
            }
            None => {
                map.set(String::new()); // Just AST errors here
                //~^ ERROR borrowed as immutable (Ast)
            }
        }
    }
}

fn err(map: &mut Map) -> &String {
    loop {
        match map.get() {
            Some(v) => {
                map.set(String::new()); // Both AST and MIR error here
                //~^ ERROR borrowed as immutable (Mir)
                //~| ERROR borrowed as immutable (Ast)
                return v;
            }
            None => {
                map.set(String::new()); // Just AST errors here
                //~^ ERROR borrowed as immutable (Ast)
            }
        }
    }
}

fn main() { }
