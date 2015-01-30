// Copyright 2014 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

#![allow(unknown_features)]
#![feature(box_syntax)]

fn match_on_local() {
    let mut foo = Some(box 5);
    match foo {
        None => {},
        Some(x) => {
            foo = Some(x);
        }
    }
    println!("'{}'", foo.unwrap());
}

fn match_on_arg(mut foo: Option<Box<i32>>) {
    match foo {
        None => {}
        Some(x) => {
            foo = Some(x);
        }
    }
    println!("'{}'", foo.unwrap());
}

fn match_on_binding() {
    match Some(box 7) {
        mut foo => {
            match foo {
                None => {},
                Some(x) => {
                    foo = Some(x);
                }
            }
            println!("'{}'", foo.unwrap());
        }
    }
}

fn match_on_upvar() {
    let mut foo = Some(box 8i32);
    let f = move|:| {
        match foo {
            None => {},
            Some(x) => {
                foo = Some(x);
            }
        }
        println!("'{}'", foo.unwrap());
    };
    f();
}

fn main() {
    match_on_local();
    match_on_arg(Some(box 6));
    match_on_binding();
    match_on_upvar();
}
