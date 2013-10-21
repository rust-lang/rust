// Copyright 2013 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

// Tests correct kind-checking of the reason stack closures without the :Copy
// bound must be noncopyable. For details see
// http://smallcultfollowing.com/babysteps/blog/2013/04/30/the-case-of-the-recurring-closure/

struct R<'self> {
    // This struct is needed to create the
    // otherwise infinite type of a fn that
    // accepts itself as argument:
    c: &'self fn(&R, bool)
}

fn innocent_looking_victim() {
    let mut x = Some(~"hello");
    do conspirator |f, writer| {
        if writer {
            x = None;
        } else {
            match x {
                Some(ref msg) => {
                    (f.c)(f, true);
                    println!("{:?}", msg);
                },
                None => fail!("oops"),
            }
        }
    }
}

fn conspirator(f: &fn(&R, bool)) {
    let r = R {c: f};
    f(&r, false) //~ ERROR use of moved value
}

fn main() { innocent_looking_victim() }
