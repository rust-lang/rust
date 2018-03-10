// Copyright 2016 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

struct Ref<'a, T: 'a> {
    data: &'a T
}

fn foo<'a>(x: Ref<i32>, y: &mut Vec<Ref<'a, i32>>) {
    y.push(x); //~ ERROR explicit lifetime
}

fn main() { }
