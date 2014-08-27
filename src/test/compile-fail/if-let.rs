// Copyright 2014 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

#![feature(macro_rules,if_let)]

fn macros() {
    macro_rules! foo{
        ($p:pat, $e:expr, $b:block) => {{
            if let $p = $e $b
        }}
    }
    macro_rules! bar{
        ($p:pat, $e:expr, $b:block) => {{
            foo!($p, $e, $b)
        }}
    }

    foo!(a, 1i, { //~ ERROR irrefutable if-let
        println!("irrefutable pattern");
    });
    bar!(a, 1i, { //~ ERROR irrefutable if-let
        println!("irrefutable pattern");
    });
}

pub fn main() {
    if let a = 1i { //~ ERROR irrefutable if-let
        println!("irrefutable pattern");
    }

    if let a = 1i { //~ ERROR irrefutable if-let
        println!("irrefutable pattern");
    } else if true {
        println!("else-if in irrefutable if-let");
    } else {
        println!("else in irrefutable if-let");
    }

    if let 1i = 2i {
        println!("refutable pattern");
    } else if let a = 1i { //~ ERROR irrefutable if-let
        println!("irrefutable pattern");
    }

    if true {
        println!("if");
    } else if let a = 1i { //~ ERROR irrefutable if-let
        println!("irrefutable pattern");
    }
}
