// Copyright 2014 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.


struct Foo {
    string: String
}

pub fn main() {
    let x = [
        Foo { string: "foo".to_strbuf() },
        Foo { string: "bar".to_strbuf() },
        Foo { string: "baz".to_strbuf() }
    ];
    match x {
        [ref first, ..tail] => {
            assert!(first.string == "foo".to_strbuf());
            assert_eq!(tail.len(), 2);
            assert!(tail[0].string == "bar".to_strbuf());
            assert!(tail[1].string == "baz".to_strbuf());

            match tail {
                [Foo { .. }, _, Foo { .. }, .. _tail] => {
                    unreachable!();
                }
                [Foo { string: ref a }, Foo { string: ref b }] => {
                    assert_eq!("bar", a.as_slice().slice(0, a.len()));
                    assert_eq!("baz", b.as_slice().slice(0, b.len()));
                }
                _ => {
                    unreachable!();
                }
            }
        }
    }
}
