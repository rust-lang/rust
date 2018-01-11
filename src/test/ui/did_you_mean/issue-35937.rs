// Copyright 2017 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

struct Foo {
    pub v: Vec<String>
}

fn main() {
    let f = Foo { v: Vec::new() };
    f.v.push("cat".to_string()); //~ ERROR cannot borrow
}


struct S {
    x: i32,
}
fn foo() {
    let s = S { x: 42 };
    s.x += 1; //~ ERROR cannot assign
}

fn bar(s: S) {
    s.x += 1; //~ ERROR cannot assign
}
