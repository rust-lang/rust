// Copyright 2017 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

enum Bar {
    Qux,
    Zar,
}

struct Foo {
    bar: usize,
}

struct X<T1, T2> {
    x: T1,
    y: T2,
}

fn a() -> Foo {
    Some(Foo { bar: 1 }) //~ ERROR mismatched types
}

fn a2() -> Foo {
    Ok(Foo { bar: 1}) //~ ERROR mismatched types
}

fn b() -> Option<Foo> {
    Foo { bar: 1 } //~ ERROR mismatched types
}

fn c() -> Result<Foo, Bar> {
    Foo { bar: 1 } //~ ERROR mismatched types
}

fn d() -> X<X<String, String>, String> {
    let x = X {
        x: X {
            x: "".to_string(),
            y: 2,
        },
        y: 3,
    };
    x //~ ERROR mismatched types
}

fn e() -> X<X<String, String>, String> {
    let x = X {
        x: X {
            x: "".to_string(),
            y: 2,
        },
        y: "".to_string(),
    };
    x //~ ERROR mismatched types
}

fn main() {}
