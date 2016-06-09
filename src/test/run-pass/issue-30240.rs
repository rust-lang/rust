// Copyright 2016 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

fn main() {
    let &ref a = &[0i32] as &[_];
    assert_eq!(a, &[0i32] as &[_]);

    let &ref a = "hello";
    assert_eq!(a, "hello");

    match "foo" {
        "fool" => unreachable!(),
        "foo" => {},
        ref _x => unreachable!()
    }
}
