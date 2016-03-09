// Copyright 2012-2014 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

#![feature(slice_patterns)]
#![feature(advanced_slice_patterns)]

enum Enum { First, Second(bool) }

fn vectors_with_nested_enums() {
    let x: &'static [Enum] = &[Enum::First, Enum::Second(false)];
    match x {
    //#^ ERROR non-exhaustive patterns: `[Second(true), Second(false)]` not covered
        [] => (),
        [_] => (),
        [Enum::First, _] => (),
        [Enum::Second(true), Enum::First] => (),
        [Enum::Second(true), Enum::Second(true)] => (),
        [Enum::Second(false), _] => (),
        [_, _, tail.., _] => ()
        //~^ ERROR slice patterns are badly broken
    }
}

fn main() {
    let vec = vec!(Some(42), None, Some(21));
    match &*vec { //# ERROR non-exhaustive patterns: `[]` not covered
        [Some(..), None, tail..] => {}
        //~^ ERROR slice patterns are badly broken
        [Some(..), Some(..), tail..] => {}
        //~^ ERROR slice patterns are badly broken
        [None] => {}
    }

    let vec = vec!(Some(42), None, Some(21));
    let vec: &[Option<isize>] = &vec;
    match vec {
        [Some(..), None, tail..] => {}
        //~^ ERROR slice patterns are badly broken
        [Some(..), Some(..), tail..] => {}
        //~^ ERROR slice patterns are badly broken
        [None, None, tail..] => {}
        //~^ ERROR slice patterns are badly broken
        [None, Some(..), tail..] => {}
        //~^ ERROR slice patterns are badly broken
        [Some(_)] => {}
        [None] => {}
        [] => {}
    }
    let vec = vec!(1);
    let vec: &[isize] = &vec;
    match vec {
        [_, tail..] => (),
        //~^ ERROR slice patterns are badly broken
        [] => ()
    }
}
