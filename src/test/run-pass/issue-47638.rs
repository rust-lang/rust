// Copyright 2018 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

fn id<'c, 'b>(f: &'c &'b Fn(&i32)) -> &'c &'b Fn(&'static i32) {
    f
}

fn main() {
    let f: &Fn(&i32) = &|x| {};
    id(&f);
}
