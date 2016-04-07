// Copyright 2014 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

// Test `?Sized` types not allowed in fields (except the last one).

struct S1<X: ?Sized> {
    f1: X, //~ ERROR `X: std::marker::Sized` is not satisfied
    f2: isize,
}
struct S2<X: ?Sized> {
    f: isize,
    g: X, //~ ERROR `X: std::marker::Sized` is not satisfied
    h: isize,
}
struct S3 {
    f: str, //~ ERROR `str: std::marker::Sized` is not satisfied
    g: [usize]
}
struct S4 {
    f: [u8], //~ ERROR `[u8]: std::marker::Sized` is not satisfied
    g: usize
}
enum E<X: ?Sized> {
    V1(X, isize), //~ERROR `X: std::marker::Sized` is not satisfied
}
enum F<X: ?Sized> {
    V2{f1: X, f: isize}, //~ERROR `X: std::marker::Sized` is not satisfied
}

pub fn main() {
}
