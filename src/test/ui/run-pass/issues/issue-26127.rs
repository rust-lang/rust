// Copyright 2015 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

trait Tr { type T; }
impl Tr for u8 { type T=(); }
struct S<I: Tr>(I::T);

fn foo<I: Tr>(i: I::T) {
    S::<I>(i);
}

fn main() {
    foo::<u8>(());
}
