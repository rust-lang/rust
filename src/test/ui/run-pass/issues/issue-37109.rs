// Copyright 2016 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

trait ToRef<'a> {
    type Ref: 'a;
}

impl<'a, U: 'a> ToRef<'a> for U {
    type Ref = &'a U;
}

fn example<'a, T>(value: &'a T) -> (<T as ToRef<'a>>::Ref, u32) {
    (value, 0)
}

fn main() {
    example(&0);
}
