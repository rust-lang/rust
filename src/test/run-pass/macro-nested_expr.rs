// Copyright 2017 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

// #42164

#![feature(decl_macro)]
#![allow(dead_code)]

pub macro m($inner_str:expr) {
    #[doc = $inner_str]
    struct S;
}

macro_rules! define_f {
    ($name:expr) => {
        #[export_name = $name]
        fn f() {}
    }
}

fn main() {
    define_f!(concat!("exported_", "f"));
    m!(stringify!(foo));
}

