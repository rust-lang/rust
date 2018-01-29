// Copyright 2012 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

// The logic for parsing Kleene operators in macros has a special case to disambiguate `?`.
// Specifically, `$(pat)?` is the ZeroOrOne operator whereas `$(pat)?+` or `$(pat)?*` are the
// ZeroOrMore and OneOrMore operators using `?` as a separator. These tests are intended to
// exercise that logic in the macro parser.
//
// Moreover, we also throw in some tests for using a separator with `?`, which is meaningless but
// included for consistency with `+` and `*`.
//
// This test focuses on non-error cases and making sure the correct number of repetitions happen.

#![feature(macro_at_most_once_rep)]

macro_rules! foo {
    ($($a:ident)? ; $num:expr) => { {
        let mut x = 0;

        $(
            x += $a;
         )?

        assert_eq!(x, $num);
    } }
}

macro_rules! baz {
    ($($a:ident),? ; $num:expr) => { { // comma separator is meaningless for `?`
        let mut x = 0;

        $(
            x += $a;
         )?

        assert_eq!(x, $num);
    } }
}

macro_rules! barplus {
    ($($a:ident)?+ ; $num:expr) => { {
        let mut x = 0;

        $(
            x += $a;
         )+

        assert_eq!(x, $num);
    } }
}

macro_rules! barstar {
    ($($a:ident)?* ; $num:expr) => { {
        let mut x = 0;

        $(
            x += $a;
         )*

        assert_eq!(x, $num);
    } }
}

pub fn main() {
    let a = 1;

    // accept 0 or 1 repetitions
    foo!( ; 0);
    foo!(a ; 1);
    baz!( ; 0);
    baz!(a ; 1);

    // Make sure using ? as a separator works as before
    barplus!(a ; 1);
    barplus!(a?a ; 2);
    barplus!(a?a?a ; 3);
    barstar!( ; 0);
    barstar!(a ; 1);
    barstar!(a?a ; 2);
    barstar!(a?a?a ; 3);
}
