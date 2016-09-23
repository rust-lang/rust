// Copyright 2015 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

// Issue #21633: reject duplicate loop labels in function bodies.
// This is testing interaction between lifetime-params and labels.

#![feature(rustc_attrs)]

#![allow(dead_code, unused_variables)]

fn foo() {
    fn foo<'a>() { //~ NOTE first declared here
        'a: loop { break 'a; }
        //~^ WARN label name `'a` shadows a lifetime name that is already in scope
        //~| NOTE lifetime 'a already in scope
    }

    struct Struct<'b, 'c> { _f: &'b i8, _g: &'c i8 }
    enum Enum<'d, 'e> { A(&'d i8), B(&'e i8) }

    impl<'d, 'e> Struct<'d, 'e> {
        fn meth_okay() {
            'a: loop { break 'a; }
            'b: loop { break 'b; }
            'c: loop { break 'c; }
        }
    }

    impl <'d, 'e> Enum<'d, 'e> {
        fn meth_okay() {
            'a: loop { break 'a; }
            'b: loop { break 'b; }
            'c: loop { break 'c; }
        }
    }

    impl<'bad, 'c> Struct<'bad, 'c> { //~ NOTE first declared here
        fn meth_bad(&self) {
            'bad: loop { break 'bad; }
            //~^ WARN label name `'bad` shadows a lifetime name that is already in scope
            //~| NOTE lifetime 'bad already in scope
        }
    }

    impl<'b, 'bad> Struct<'b, 'bad> { //~ NOTE first declared here
        fn meth_bad2(&self) {
            'bad: loop { break 'bad; }
            //~^ WARN label name `'bad` shadows a lifetime name that is already in scope
            //~| NOTE lifetime 'bad already in scope
        }
    }

    impl<'b, 'c> Struct<'b, 'c> {
        fn meth_bad3<'bad>(x: &'bad i8) { //~ NOTE first declared here
            'bad: loop { break 'bad; }
            //~^ WARN label name `'bad` shadows a lifetime name that is already in scope
            //~| NOTE lifetime 'bad already in scope
        }

        fn meth_bad4<'a,'bad>(x: &'a i8, y: &'bad i8) {
            //~^ NOTE first declared here
            'bad: loop { break 'bad; }
            //~^ WARN label name `'bad` shadows a lifetime name that is already in scope
            //~| NOTE lifetime 'bad already in scope
        }
    }

    impl <'bad, 'e> Enum<'bad, 'e> { //~ NOTE first declared here
        fn meth_bad(&self) {
            'bad: loop { break 'bad; }
            //~^ WARN label name `'bad` shadows a lifetime name that is already in scope
            //~| NOTE lifetime 'bad already in scope
        }
    }
    impl <'d, 'bad> Enum<'d, 'bad> { //~ NOTE first declared here
        fn meth_bad2(&self) {
            'bad: loop { break 'bad; }
            //~^ WARN label name `'bad` shadows a lifetime name that is already in scope
            //~| NOTE lifetime 'bad already in scope
        }
    }
    impl <'d, 'e> Enum<'d, 'e> {
        fn meth_bad3<'bad>(x: &'bad i8) { //~ NOTE first declared here
            'bad: loop { break 'bad; }
            //~^ WARN label name `'bad` shadows a lifetime name that is already in scope
            //~| NOTE lifetime 'bad already in scope
        }

        fn meth_bad4<'a,'bad>(x: &'bad i8) { //~ NOTE first declared here
            'bad: loop { break 'bad; }
            //~^ WARN label name `'bad` shadows a lifetime name that is already in scope
            //~| NOTE lifetime 'bad already in scope
        }
    }

    trait HasDefaultMethod1<'bad> { //~ NOTE first declared here
        fn meth_okay() {
            'c: loop { break 'c; }
        }
        fn meth_bad(&self) {
            'bad: loop { break 'bad; }
            //~^ WARN label name `'bad` shadows a lifetime name that is already in scope
            //~| NOTE lifetime 'bad already in scope
        }
    }
    trait HasDefaultMethod2<'a,'bad> { //~ NOTE first declared here
        fn meth_bad(&self) {
            'bad: loop { break 'bad; }
            //~^ WARN label name `'bad` shadows a lifetime name that is already in scope
            //~| NOTE lifetime 'bad already in scope
        }
    }
    trait HasDefaultMethod3<'a,'b> {
        fn meth_bad<'bad>(&self) { //~ NOTE first declared here
            'bad: loop { break 'bad; }
            //~^ WARN label name `'bad` shadows a lifetime name that is already in scope
            //~| NOTE lifetime 'bad already in scope
        }
    }
}

#[rustc_error]
pub fn main() { //~ ERROR compilation successful
    foo();
}
