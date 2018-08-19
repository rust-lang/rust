// Copyright 2018 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

macro_rules! foo {
    ($a:ident) => ();
    ($a:ident, $b:ident) => ();
    ($a:ident, $b:ident, $c:ident) => ();
    ($a:ident, $b:ident, $c:ident, $d:ident) => ();
    ($a:ident, $b:ident, $c:ident, $d:ident, $e:ident) => ();
}

fn main() {
    println!("{}" a);
    //~^ ERROR expected token: `,`
    foo!(a b);
    //~^ ERROR no rules expected the token `b`
    foo!(a, b, c, d e);
    //~^ ERROR no rules expected the token `e`
    foo!(a, b, c d, e);
    //~^ ERROR no rules expected the token `d`
    foo!(a, b, c d e);
    //~^ ERROR no rules expected the token `d`
}
