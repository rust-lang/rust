// Copyright 2012 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

#[feature(managed_boxes)];

trait T {
    fn print(&self);
}

struct S {
    s: int,
}

impl T for S {
    fn print(&self) {
        println!("{:?}", self);
    }
}

fn print_t(t: &T) {
    t.print();
}

fn print_s(s: &S) {
    s.print();
}

pub fn main() {
    let s: @S = @S { s: 5 };
    print_s(s);
    let t: @T = s as @T;
    print_t(t);

}
