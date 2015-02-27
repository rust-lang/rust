#![feature(no_std)]
#![no_std]
#[prelude_import]
use std::prelude::v1::*;
#[macro_use]
extern crate "std" as std;
// Copyright 2014 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

// pretty-compare-only
// pretty-mode:typed
// pp-exact:issue-4264.pp

// #4264 fixed-length vector types

pub fn foo(_: [i32; (3: usize)]) { }

pub fn bar() {
    const FOO: usize = ((5: usize) - (4: usize): usize);
    let _: [(); (FOO: usize)] = ([((): ())]: [(); 1]);

    let _: [(); (1usize: usize)] = ([((): ())]: [(); 1]);

    let _ =
        (((&((([(1: i32), (2: i32), (3: i32)]: [i32; 3])): [i32; 3]):
              &[i32; 3]) as *const _: *const [i32; 3]) as
            *const [i32; (3usize: usize)]: *const [i32; 3]);









    ((::std::fmt::format:
         fn(core::fmt::Arguments<'_>) -> collections::string::String {collections::fmt::format})(((::std::fmt::Arguments::new_v1:
                                                                                                      fn(&[&str], &[core::fmt::ArgumentV1<'_>]) -> core::fmt::Arguments<'_> {core::fmt::Arguments<'a>::new_v1})(({
                                                                                                                                                                                                                     static __STATIC_FMTSTR:
                                                                                                                                                                                                                            &'static [&'static str]
                                                                                                                                                                                                                            =
                                                                                                                                                                                                                         (&([("test":
                                                                                                                                                                                                                                 &'static str)]:
                                                                                                                                                                                                                               [&'static str; 1]):
                                                                                                                                                                                                                             &'static [&'static str; 1]);
                                                                                                                                                                                                                     (__STATIC_FMTSTR:
                                                                                                                                                                                                                         &'static [&'static str])
                                                                                                                                                                                                                 }:
                                                                                                                                                                                                                    &[&str]),
                                                                                                                                                                                                                (&(match (():
                                                                                                                                                                                                                             ())
                                                                                                                                                                                                                       {
                                                                                                                                                                                                                       ()
                                                                                                                                                                                                                       =>
                                                                                                                                                                                                                       ([]:
                                                                                                                                                                                                                           [core::fmt::ArgumentV1<'_>; 0]),
                                                                                                                                                                                                                   }:
                                                                                                                                                                                                                      [core::fmt::ArgumentV1<'_>; 0]):
                                                                                                                                                                                                                    &[core::fmt::ArgumentV1<'_>; 0])):
                                                                                                     core::fmt::Arguments<'_>)):
        collections::string::String);
}
pub type Foo = [i32; (3: usize)];
pub struct Bar {
    pub x: [i32; (3: usize)],
}
pub struct TupleBar([i32; (4: usize)]);
pub enum Baz { BazVariant([i32; (5: usize)]), }
pub fn id<T>(x: T) -> T { (x: T) }
pub fn use_id() {
    let _ =
        ((id::<[i32; (3: usize)]>:
             fn([i32; 3]) -> [i32; 3] {id})(([(1: i32), (2: i32), (3: i32)]:
                                                [i32; 3])): [i32; 3]);
}
fn main() { }
