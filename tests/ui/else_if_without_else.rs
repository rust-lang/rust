// Copyright 2014-2018 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

#![warn(clippy::all)]
#![warn(clippy::else_if_without_else)]

fn bla1() -> bool {
    unimplemented!()
}
fn bla2() -> bool {
    unimplemented!()
}
fn bla3() -> bool {
    unimplemented!()
}

fn main() {
    if bla1() {
        println!("if");
    }

    if bla1() {
        println!("if");
    } else {
        println!("else");
    }

    if bla1() {
        println!("if");
    } else if bla2() {
        println!("else if");
    } else {
        println!("else")
    }

    if bla1() {
        println!("if");
    } else if bla2() {
        println!("else if 1");
    } else if bla3() {
        println!("else if 2");
    } else {
        println!("else")
    }

    if bla1() {
        println!("if");
    } else if bla2() {
        //~ ERROR else if without else
        println!("else if");
    }

    if bla1() {
        println!("if");
    } else if bla2() {
        println!("else if 1");
    } else if bla3() {
        //~ ERROR else if without else
        println!("else if 2");
    }
}
