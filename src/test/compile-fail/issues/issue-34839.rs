// Copyright 2016 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

#![feature(rustc_attrs)]
#![allow(dead_code)]

trait RegularExpression: Sized {
    type Text;
}

struct ExecNoSyncStr<'a>(&'a u8);

impl<'c> RegularExpression for ExecNoSyncStr<'c> {
    type Text = u8;
}

struct FindCaptures<'t, R>(&'t R::Text) where R: RegularExpression, R::Text: 't;

enum FindCapturesInner<'r, 't> {
    Dynamic(FindCaptures<'t, ExecNoSyncStr<'r>>),
}

#[rustc_error]
fn main() {}    //~ ERROR compilation successful
