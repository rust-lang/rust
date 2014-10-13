// Copyright 2014 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

#![feature(macro_rules,while_let)]

fn macros() {
    macro_rules! foo{
        ($p:pat, $e:expr, $b:block) => {{
            while let $p = $e $b
        }}
    }
    macro_rules! bar{
        ($p:pat, $e:expr, $b:block) => {{
            foo!($p, $e, $b)
        }}
    }

    foo!(a, 1i, { //~ ERROR irrefutable while-let
        println!("irrefutable pattern");
    });
    bar!(a, 1i, { //~ ERROR irrefutable while-let
        println!("irrefutable pattern");
    });
}

pub fn main() {
    while let a = 1i { //~ ERROR irrefutable while-let
        println!("irrefutable pattern");
    }
}
