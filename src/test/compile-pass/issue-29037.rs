// Copyright 2015 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

// This test ensures that each pointer type `P<X>` is covariant in `X`.

use std::rc::Rc;
use std::sync::Arc;

fn a<'r>(x: Box<&'static str>) -> Box<&'r str> {
    x
}

fn b<'r, 'w>(x: &'w &'static str) -> &'w &'r str {
    x
}

fn c<'r>(x: Arc<&'static str>) -> Arc<&'r str> {
    x
}

fn d<'r>(x: Rc<&'static str>) -> Rc<&'r str> {
    x
}

fn main() {}
