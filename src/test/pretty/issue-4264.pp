#![no_std]
#[macro_use]
extern crate "std" as std;
#[prelude_import]
use std::prelude::v1::*;
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

pub fn foo(_: [isize; (3 as usize)]) { }

pub fn bar() {
    const FOO: usize = ((5us as usize) - (4us as usize) as usize);
    let _: [(); (FOO as usize)] = ([(() as ())] as [(); 1]);

    let _: [(); (1us as usize)] = ([(() as ())] as [(); 1]);

    let _ =
        (((&((([(1is as isize), (2 as isize), (3 as isize)] as [isize; 3])) as
                [isize; 3]) as &[isize; 3]) as *const _ as *const [isize; 3])
            as *const [isize; (3us as usize)] as *const [isize; 3]);









    ((::std::fmt::format as
         fn(core::fmt::Arguments<'_>) -> collections::string::String {std::fmt::format})(((::std::fmt::Arguments::new
                                                                                              as
                                                                                              fn(&[&str], &[core::fmt::Argument<'_>]) -> core::fmt::Arguments<'_> {core::fmt::Arguments<'a>::new})(({
                                                                                                                                                                                                        #[inline]
                                                                                                                                                                                                        #[allow(dead_code)]
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
                                                                                                                                                                                                              [core::fmt::Argument<'_>; 0]),
                                                                                                                                                                                                      }
                                                                                                                                                                                                         as
                                                                                                                                                                                                         [core::fmt::Argument<'_>; 0])
                                                                                                                                                                                                       as
                                                                                                                                                                                                       &[core::fmt::Argument<'_>; 0]))
                                                                                             as
                                                                                             core::fmt::Arguments<'_>))
        as collections::string::String);
}
pub type Foo = [isize; (3us as usize)];
pub struct Bar {
    pub x: [isize; (3us as usize)],
}
pub struct TupleBar([isize; (4us as usize)]);
pub enum Baz { BazVariant([isize; (5us as usize)]), }
pub fn id<T>(x: T) -> T { (x as T) }
pub fn use_id() {
    let _ =
        ((id::<[isize; (3us as usize)]> as
             fn([isize; 3]) -> [isize; 3] {id})(([(1 as isize), (2 as isize),
                                                  (3 as isize)] as
                                                    [isize; 3])) as
            [isize; 3]);
}
fn main() { }
