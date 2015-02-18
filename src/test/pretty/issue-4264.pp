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

pub fn foo(_: [i32; (3 as usize)]) { }

pub fn bar() {
    const FOO: usize = ((5 as usize) - (4 as usize) as usize);
    let _: [(); (FOO as usize)] = ([(() as ())] as [(); 1]);

    let _: [(); (1usize as usize)] = ([(() as ())] as [(); 1]);

    let _ =
        (((&((([(1 as i32), (2 as i32), (3 as i32)] as [i32; 3])) as [i32; 3])
              as &[i32; 3]) as *const _ as *const [i32; 3]) as
            *const [i32; (3usize as usize)] as *const [i32; 3]);









    ((::std::fmt::format as
         fn(core::fmt::Arguments<'_>) -> collections::string::String {collections::fmt::format})(((::std::fmt::Arguments::new_v1
                                                                                                      as
                                                                                                      fn(&[&str], &[core::fmt::ArgumentV1<'_>]) -> core::fmt::Arguments<'_> {core::fmt::Arguments<'a>::new_v1})(({
                                                                                                                                                                                                                     static __STATIC_FMTSTR:
                                                                                                                                                                                                                            &'static [&'static str]
                                                                                                                                                                                                                            =
                                                                                                                                                                                                                         (&([("test"
                                                                                                                                                                                                                                 as
                                                                                                                                                                                                                                 &'static str)]
                                                                                                                                                                                                                               as
                                                                                                                                                                                                                               [&'static str; 1])
                                                                                                                                                                                                                             as
                                                                                                                                                                                                                             &'static [&'static str; 1]);
                                                                                                                                                                                                                     (__STATIC_FMTSTR
                                                                                                                                                                                                                         as
                                                                                                                                                                                                                         &'static [&'static str])
                                                                                                                                                                                                                 }
                                                                                                                                                                                                                    as
                                                                                                                                                                                                                    &[&str]),
                                                                                                                                                                                                                (&(match (()
                                                                                                                                                                                                                             as
                                                                                                                                                                                                                             ())
                                                                                                                                                                                                                       {
                                                                                                                                                                                                                       ()
                                                                                                                                                                                                                       =>
                                                                                                                                                                                                                       ([]
                                                                                                                                                                                                                           as
                                                                                                                                                                                                                           [core::fmt::ArgumentV1<'_>; 0]),
                                                                                                                                                                                                                   }
                                                                                                                                                                                                                      as
                                                                                                                                                                                                                      [core::fmt::ArgumentV1<'_>; 0])
                                                                                                                                                                                                                    as
                                                                                                                                                                                                                    &[core::fmt::ArgumentV1<'_>; 0]))
                                                                                                     as
                                                                                                     core::fmt::Arguments<'_>))
        as collections::string::String);
}
pub type Foo = [i32; (3 as usize)];
pub struct Bar {
    pub x: [i32; (3 as usize)],
}
pub struct TupleBar([i32; (4 as usize)]);
pub enum Baz { BazVariant([i32; (5 as usize)]), }
pub fn id<T>(x: T) -> T { (x as T) }
pub fn use_id() {
    let _ =
        ((id::<[i32; (3 as usize)]> as
             fn([i32; 3]) -> [i32; 3] {id})(([(1 as i32), (2 as i32),
                                              (3 as i32)] as [i32; 3])) as
            [i32; 3]);
}
fn main() { }
