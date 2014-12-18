// Copyright 2014 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

// aux-build:issue-16822.rs

extern crate "issue-16822" as lib;

use std::cell::RefCell;

struct App {
    i: int
}

impl lib::Update for App {
    fn update(&mut self) {
        self.i += 1;
    }
}

fn main(){
    let app = App { i: 5 };
    let window = lib::Window { data: RefCell::new(app) };
    window.update(1);
}
