// Copyright 2014 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

struct S {
    o: Option<String>
}

// Make sure we don't reuse the same alloca when matching
// on field of struct or tuple which we reassign in the match body.

fn main() {
    let mut a = (0i, Some("right".into_string()));
    let b = match a.1 {
        Some(v) => {
            a.1 = Some("wrong".into_string());
            v
        }
        None => String::new()
    };
    println!("{}", b);
    assert_eq!(b, "right");


    let mut s = S{ o: Some("right".into_string()) };
    let b = match s.o {
        Some(v) => {
            s.o = Some("wrong".into_string());
            v
        }
        None => String::new(),
    };
    println!("{}", b);
    assert_eq!(b, "right");
}
