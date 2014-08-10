// Copyright 2014 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

fn match_on_local() {
    let mut foo = Some(box 5i);
    match foo {
        None => {},
        Some(x) => {
            foo = Some(x);
        }
    }
    println!("'{}'", foo.unwrap());
}

fn match_on_arg(mut foo: Option<Box<int>>) {
    match foo {
        None => {}
        Some(x) => {
            foo = Some(x);
        }
    }
    println!("'{}'", foo.unwrap());
}

fn match_on_binding() {
    match Some(box 7i) {
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
    let mut foo = Some(box 8i);
    (proc() {
        match foo {
            None => {},
            Some(x) => {
                foo = Some(x);
            }
        }
        println!("'{}'", foo.unwrap());
    })();
}

fn main() {
    match_on_local();
    match_on_arg(Some(box 6i));
    match_on_binding();
    match_on_upvar();
}
