// Copyright 2015 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

use foo::{x, y as fooy};
use Maybe::{Yes as MaybeYes};

pub enum Maybe { Yes, No }
mod foo {
    use super::Maybe::{self as MaybeFoo};
    pub fn x(a: MaybeFoo) {}
    pub fn y(a: i32) { println!("{}", a); }
}

pub fn main() { x(MaybeYes); fooy(10); }
