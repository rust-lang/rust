// Copyright 2014 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

use std::default::Default;

#[deriving(Default,Eq,Show)]
struct Struct {
    #[default="3"]
    x: int,
    #[default=r#""test""#]
    s: &'static str,
    #[default="primes()"]
    p: Vec<uint>,
    #[default="factorial(5)"]
    f: uint,
    child: Child,
    #[default=r#"format!("n: {}", 42)"#]
    s2: ~str
}

#[deriving(Default,Eq,Show)]
struct Tuple(
    #[default="42"]
    int,
    #[default=r#""test2""#]
    &'static str,
    #[default="primes()"]
    Vec<uint>,
    #[default="factorial(5)"]
    uint,
    Child,
    #[default=r#"format!("n: {}", 42)"#]
    ~str
);

#[deriving(Eq,Show)]
struct Child {
    name: &'static str
}

impl Default for Child {
    fn default() -> Child {
        Child { name: "child" }
    }
}

fn primes() -> Vec<uint> {
    vec![2u, 3, 5, 7, 11, 13, 17, 23]
}

fn factorial(n: uint) -> uint {
    match n {
        0|1 => 1,
        n => n*factorial(n-1)
    }
}

fn main() {
    let s: Struct = Default::default();
    assert_eq!(s, Struct {
        x: 3,
        s: "test",
        p: primes(),
        f: 120,
        child: Child { name: "child" },
        s2: "n: 42".to_owned()
    });
    let t: Tuple = Default::default();
    assert_eq!(t, Tuple(
        42,
        "test2",
        primes(),
        120,
        Child { name: "child" },
        "n: 42".to_owned()
    ));
}
