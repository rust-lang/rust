// Copyright 2017 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

#![feature(decl_macro)]

macro n($foo:ident, $S:ident, $i:ident, $m:ident) {
    mod $foo {
        #[derive(Default)]
        pub struct $S { $i: u32 }
        pub macro $m($e:expr) { $e.$i }
    }
}

n!(foo, S, i, m);

fn main() {
    use foo::{S, m};
    S::default().i; //~ ERROR field `i` of struct `foo::S` is private
    m!(S::default()); // ok
}
