#![feature(phase)]
#![no_std]
#![feature(globs)]
#[phase(plugin, link)]
extern crate "std" as std;
#[prelude_import]
use std::prelude::*;
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

pub fn foo(_: [int; (3 as uint)]) { }

pub fn bar() {
    const FOO: uint = ((5u as uint) - (4u as uint) as uint);
    let _: [(); (FOO as uint)] = ([(() as ())] as [(); 1]);

    let _: [(); (1u as uint)] = ([(() as ())] as [(); 1]);

    let _ =
        (((&((([(1i as int), (2 as int), (3 as int)] as [int; 3])) as
                [int; 3]) as &[int; 3]) as *const _ as *const [int; 3]) as
            *const [int; (3u as uint)] as *const [int; 3]);

    (match (() as ()) {
         () => {
             #[inline]
             #[allow(dead_code)]
             static __STATIC_FMTSTR: &'static [&'static str] =
                 (&([("test" as &'static str)] as [&'static str; 1]) as
                     &'static [&'static str; 1]);








             ((::std::fmt::format as
                  fn(&core::fmt::Arguments<'_>) -> collections::string::String {std::fmt::format})((&((::std::fmt::Arguments::new
                                                                                                          as
                                                                                                          fn(&[&str], &[core::fmt::Argument<'_>]) -> core::fmt::Arguments<'_> {core::fmt::Arguments<'a>::new})((__STATIC_FMTSTR
                                                                                                                                                                                                                   as
                                                                                                                                                                                                                   &'static [&'static str]),
                                                                                                                                                                                                               (&([]
                                                                                                                                                                                                                     as
                                                                                                                                                                                                                     [core::fmt::Argument<'_>; 0])
                                                                                                                                                                                                                   as
                                                                                                                                                                                                                   &[core::fmt::Argument<'_>; 0]))
                                                                                                         as
                                                                                                         core::fmt::Arguments<'_>)
                                                                                                       as
                                                                                                       &core::fmt::Arguments<'_>))
                 as collections::string::String)
         }
     } as collections::string::String);
}
pub type Foo = [int; (3u as uint)];
pub struct Bar {
    pub x: [int; (3u as uint)],
}
pub struct TupleBar([int; (4u as uint)]);
pub enum Baz { BazVariant([int; (5u as uint)]), }
pub fn id<T>(x: T) -> T { (x as T) }
pub fn use_id() {
    let _ =
        ((id::<[int; (3u as uint)]> as
             fn([int; 3]) -> [int; 3] {id})(([(1 as int), (2 as int),
                                              (3 as int)] as [int; 3])) as
            [int; 3]);
}
fn main() { }
