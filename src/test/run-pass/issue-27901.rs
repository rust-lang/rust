// Copyright 2017 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

trait Stream { type Item; }
impl<'a> Stream for &'a str { type Item = u8; }
fn f<'s>(s: &'s str) -> (&'s str, <&'s str as Stream>::Item) {
    (s, 42)
}

fn main() {
    let fx = f as for<'t> fn(&'t str) -> (&'t str, <&'t str as Stream>::Item);
    assert_eq!(fx("hi"), ("hi", 42));
}
