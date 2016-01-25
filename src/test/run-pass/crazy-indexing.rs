// Copyright 201&6 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

#![feature(collections_range)]

use std::ops::RangeArgument;

struct Idx<'a>(Option<&'a usize>, Option<&'a usize>);

impl<'a> RangeArgument<usize> for Idx<'a> {
    fn start(&self) -> Option<&usize> {
        self.0
    }

    fn end(&self) -> Option<&usize> {
        self.1
    }
}

#[cfg(not(windows))]
fn windows() {}

#[cfg(windows)]
fn windows() {
    use std::sys::common::wtf8::Wtf8Buf;

    let wtf = Wtf8Buf::from_str("hello world");

    assert_eq!(&wtf[Idx(Some(&3), Some(&6))], &wtf[3..6]);
    assert_eq!(&wtf[Idx(None    , None    )], &wtf[..]);
    assert_eq!(&wtf[Idx(Some(&3), None    )], &wtf[3..]);
    assert_eq!(&wtf[Idx(None    , Some(&6))], &wtf[..6]);
}

fn main() {
    let slice: &[_] = &*(0..10).collect::<Vec<_>>();
    let string = String::from("hello world");
    let stir: &str = &*string;

    assert_eq!(&slice[Idx(Some(&3), Some(&6))], &slice[3..6]);
    assert_eq!(&slice[Idx(None    , None    )], &slice[..]);
    assert_eq!(&slice[Idx(Some(&3), None    )], &slice[3..]);
    assert_eq!(&slice[Idx(None    , Some(&6))], &slice[..6]);

    assert_eq!(&string[Idx(Some(&3), Some(&6))], &string[3..6]);
    assert_eq!(&string[Idx(None    , None    )], &string[..]);
    assert_eq!(&string[Idx(Some(&3), None    )], &string[3..]);
    assert_eq!(&string[Idx(None    , Some(&6))], &string[..6]);

    assert_eq!(&stir[Idx(Some(&3), Some(&6))], &stir[3..6]);
    assert_eq!(&stir[Idx(None    , None    )], &stir[..]);
    assert_eq!(&stir[Idx(Some(&3), None    )], &stir[3..]);
    assert_eq!(&stir[Idx(None    , Some(&6))], &stir[..6]);

    windows();
}

