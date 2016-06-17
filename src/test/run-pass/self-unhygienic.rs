// Copyright 2016 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

// Check that `self` is unhygienic

struct A {
    pub v: i64
}

macro_rules! pretty {
    ( $t:ty, $field:ident ) => (
        impl $t {
            pub fn pretty(&self) -> String {
                format!("<A:{}>", self.$field)
            }
        }
    )
}

pretty!(A, v);

fn main() {
    assert_eq!(format!("Pretty: {}", A { v: 3 }.pretty()), "Pretty: <A:3>");
    println!("{}", A{ v: 42 }.pretty());
}
