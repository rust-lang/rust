// Copyright 2017 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

// ignore-pretty pretty-printing is unhygienic

#![feature(decl_macro)]
#![allow(unused)]

mod foo {
    pub macro m($s:tt, $i:tt) {
        $s.$i
    }
}

mod bar {
    struct S(i32);
    fn f() {
        let s = S(0);
        ::foo::m!(s, 0);
    }
}

fn main() {}
