// Copyright 2012 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.




type t = int;

fn nothing() { }

fn putstr(_s: String) { }

fn putint(_i: int) {
    let mut i: int = 33;
    while i < 36 { putstr("hi".to_string()); i = i + 1; }
}

fn zerg(i: int) -> int { return i; }

fn foo(x: int) -> int {
    let mut y: t = x + 2;
    putstr("hello".to_string());
    while y < 10 { putint(y); if y * 3 == 4 { y = y + 2; nothing(); } }
    let mut z: t;
    z = 0x55;
    foo(z);
    return 0;
}

pub fn main() {
    let x: int = 2i + 2;
    println!("{}", x);
    println!("hello, world");
    println!("{}", 10i);
}
