// Copyright 2014 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

// ignore-fast

// Check we do the correct privacy checks when we import a name and there is an
// item with that name in both the value and type namespaces.

#[feature(globs)];
#[allow(dead_code)];
#[allow(unused_imports)];

// public type, private value
pub mod foo1 {
    pub trait Bar {
    }
    pub struct Baz;

    fn Bar() { }
}

fn test_unused1() {
    use foo1::Bar;
    use foo1::{Bar,Baz};
    use foo1::*;
}

fn test_single1() {
    use foo1::Bar;

    let _x: ~Bar;
}

fn test_list1() {
    use foo1::{Bar,Baz};

    let _x: ~Bar;
}

fn test_glob1() {
    use foo1::*;

    let _x: ~Bar;
}

// private type, public value
pub mod foo2 {
    trait Bar {
    }
    pub struct Baz;

    pub fn Bar() { }
}

fn test_unused2() {
    use foo2::Bar;
    use foo2::{Bar,Baz};
    use foo2::*;
}

fn test_single2() {
    use foo2::Bar;

    Bar();
}

fn test_list2() {
    use foo2::{Bar,Baz};

    Bar();
}

fn test_glob2() {
    use foo2::*;

    Bar();
}

// public type, public value
pub mod foo3 {
    pub trait Bar {
    }
    pub struct Baz;

    pub fn Bar() { }
}

fn test_unused3() {
    use foo3::Bar;
    use foo3::{Bar,Baz};
    use foo3::*;
}

fn test_single3() {
    use foo3::Bar;

    Bar();
    let _x: ~Bar;
}

fn test_list3() {
    use foo3::{Bar,Baz};

    Bar();
    let _x: ~Bar;
}

fn test_glob3() {
    use foo3::*;

    Bar();
    let _x: ~Bar;
}

fn main() {
}

