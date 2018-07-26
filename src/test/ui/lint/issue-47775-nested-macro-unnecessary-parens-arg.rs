// Copyright 2018 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

// compile-pass

#![warn(unused_parens)]

macro_rules! the_worship_the_heart_lifts_above {
    ( @as_expr, $e:expr) => { $e };
    ( @generate_fn, $name:tt) => {
        #[allow(dead_code)] fn the_moth_for_the_star<'a>() -> Option<&'a str> {
            Some(the_worship_the_heart_lifts_above!( @as_expr, $name ))
        }
    };
    ( $name:ident ) => { the_worship_the_heart_lifts_above!( @generate_fn, (stringify!($name))); }
    // ↑ Notably, this does 𝘯𝘰𝘵 warn: we're declining to lint unused parens in
    // function/method arguments inside of nested macros because of situations
    // like those reported in Issue #47775
}

macro_rules! and_the_heavens_reject_not {
    () => {
        // ↓ But let's test that we still lint for unused parens around
        // function args inside of simple, one-deep macros.
        #[allow(dead_code)] fn the_night_for_the_morrow() -> Option<isize> { Some((2)) }
        //~^ WARN unnecessary parentheses around function argument
    }
}

the_worship_the_heart_lifts_above!(rah);
and_the_heavens_reject_not!();

fn main() {}
