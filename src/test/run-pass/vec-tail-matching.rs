// Copyright 2014 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.



#![feature(slice_patterns)]

struct Foo {
    string: &'static str
}

pub fn main() {
    let x = [
        Foo { string: "foo" },
        Foo { string: "bar" },
        Foo { string: "baz" }
    ];
    match x {
        [ref first, ref tail..] => {
            assert_eq!(first.string, "foo");
            assert_eq!(tail.len(), 2);
            assert_eq!(tail[0].string, "bar");
            assert_eq!(tail[1].string, "baz");

            match *(tail as &[_]) {
                [Foo { .. }, _, Foo { .. }, ref _tail..] => {
                    unreachable!();
                }
                [Foo { string: ref a }, Foo { string: ref b }] => {
                    assert_eq!("bar", &a[0..a.len()]);
                    assert_eq!("baz", &b[0..b.len()]);
                }
                _ => {
                    unreachable!();
                }
            }
        }
    }
}
