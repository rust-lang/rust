// xfail-fast

// Copyright 2013 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

// xfail-test FIXME(#5121)

#[feature(managed_boxes)];

extern mod extra;

// These tests used to be separate files, but I wanted to refactor all
// the common code.

use std::hashmap::{HashMap, HashSet};

use EBReader = extra::ebml::reader;
use EBWriter = extra::ebml::writer;
use std::cmp::Eq;
use std::cmp;
use std::io;
use extra::serialize::{Decodable, Encodable};
use extra::time;

fn test_ebml<'a, A:
    Eq +
    Encodable<EBWriter::Encoder> +
    Decodable<EBReader::Decoder<'a>>
>(a1: &A) {
    let mut wr = std::io::MemWriter::new();
    let mut ebml_w = EBWriter::Encoder(&mut wr);
    a1.encode(&mut ebml_w);
    let bytes = wr.get_ref();

    let d: extra::ebml::Doc<'a> = EBReader::Doc(bytes);
    let mut decoder: EBReader::Decoder<'a> = EBReader::Decoder(d);
    let a2: A = Decodable::decode(&mut decoder);
    assert!(*a1 == a2);
}

#[deriving(Decodable, Encodable)]
enum Expr {
    Val(uint),
    Plus(@Expr, @Expr),
    Minus(@Expr, @Expr)
}

impl cmp::Eq for Expr {
    fn eq(&self, other: &Expr) -> bool {
        match *self {
            Val(e0a) => {
                match *other {
                    Val(e0b) => e0a == e0b,
                    _ => false
                }
            }
            Plus(e0a, e1a) => {
                match *other {
                    Plus(e0b, e1b) => e0a == e0b && e1a == e1b,
                    _ => false
                }
            }
            Minus(e0a, e1a) => {
                match *other {
                    Minus(e0b, e1b) => e0a == e0b && e1a == e1b,
                    _ => false
                }
            }
        }
    }
    fn ne(&self, other: &Expr) -> bool { !(*self).eq(other) }
}

impl cmp::Eq for Point {
    fn eq(&self, other: &Point) -> bool {
        self.x == other.x && self.y == other.y
    }
    fn ne(&self, other: &Point) -> bool { !(*self).eq(other) }
}

impl<T:cmp::Eq> cmp::Eq for Quark<T> {
    fn eq(&self, other: &Quark<T>) -> bool {
        match *self {
            Top(ref q) => {
                match *other {
                    Top(ref r) => q == r,
                    Bottom(_) => false
                }
            },
            Bottom(ref q) => {
                match *other {
                    Top(_) => false,
                    Bottom(ref r) => q == r
                }
            },
        }
    }
    fn ne(&self, other: &Quark<T>) -> bool { !(*self).eq(other) }
}

impl cmp::Eq for CLike {
    fn eq(&self, other: &CLike) -> bool {
        (*self) as int == *other as int
    }
    fn ne(&self, other: &CLike) -> bool { !self.eq(other) }
}

#[deriving(Decodable, Encodable, Eq)]
struct Spanned<T> {
    lo: uint,
    hi: uint,
    node: T,
}

#[deriving(Decodable, Encodable)]
struct SomeStruct { v: ~[uint] }

#[deriving(Decodable, Encodable)]
struct Point {x: uint, y: uint}

#[deriving(Decodable, Encodable)]
enum Quark<T> {
    Top(T),
    Bottom(T)
}

#[deriving(Decodable, Encodable)]
enum CLike { A, B, C }

pub fn main() {
    let a = &Plus(@Minus(@Val(3u), @Val(10u)), @Plus(@Val(22u), @Val(5u)));
    test_ebml(a);

    let a = &Spanned {lo: 0u, hi: 5u, node: 22u};
    test_ebml(a);

    let a = &Point {x: 3u, y: 5u};
    test_ebml(a);

    let a = &Top(22u);
    test_ebml(a);

    let a = &Bottom(222u);
    test_ebml(a);

    let a = &A;
    test_ebml(a);

    let a = &B;
    test_ebml(a);

    let a = &time::now();
    test_ebml(a);

    test_ebml(&1.0f32);
    test_ebml(&1.0f64);
    test_ebml(&'a');

    let mut a = HashMap::new();
    test_ebml(&a);
    a.insert(1, 2);
    test_ebml(&a);

    let mut a = HashSet::new();
    test_ebml(&a);
    a.insert(1);
    test_ebml(&a);
}
