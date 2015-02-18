// Copyright 2014 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

#![feature(box_syntax)]

fn main() {
    let _foo = &[1_usize, 2] as [usize];
    //~^ ERROR cast to unsized type: `&[usize; 2]` as `[usize]`
    //~^^ HELP consider using an implicit coercion to `&[usize]` instead
    let _bar = box 1_usize as std::fmt::Show;
    //~^ ERROR cast to unsized type: `Box<usize>` as `core::fmt::Show`
    //~^^ HELP did you mean `Box<core::fmt::Show>`?
    let _baz = 1_usize as std::fmt::Show;
    //~^ ERROR cast to unsized type: `usize` as `core::fmt::Show`
    //~^^ HELP consider using a box or reference as appropriate
    let _quux = [1_usize, 2] as [usize];
    //~^ ERROR cast to unsized type: `[usize; 2]` as `[usize]`
    //~^^ HELP consider using a box or reference as appropriate
}
