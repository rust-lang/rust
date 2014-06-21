// Copyright 2014 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

fn foo<T: Add<T, T> + Clone>([x, y, z]: [T, ..3]) -> (T, T, T) {
    (x.clone(), x.clone() + y.clone(), x + y + z)
}
fn bar(a: &'static str, b: &'static str) -> [&'static str, ..4] {
    [a, b, b, a]
}

fn main() {
    assert_eq!(foo([1, 2, 3]), (1, 3, 6));

    let [a, b, c, d] = bar("foo", "bar");
    assert_eq!(a, "foo");
    assert_eq!(b, "bar");
    assert_eq!(c, "bar");
    assert_eq!(d, "foo");

    let [a, _, _, d] = bar("baz", "foo");
    assert_eq!(a, "baz");
    assert_eq!(d, "baz");

    let out = bar("baz", "foo");
    let [a, ..xs, d] = out;
    assert_eq!(a, "baz");
    assert!(xs == ["foo", "foo"]);
    assert_eq!(d, "baz");
}