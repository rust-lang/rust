// Copyright 2014 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.


//! https://github.com/rust-lang/rust/pull/20824

// ignore-pretty

macro_rules! matches {
    ($expression: expr, $($pattern:pat)|+) => {
        matches!($expression, $($pattern)|+ if true)
    };
    ($expression: expr, $($pattern:pat)|+ if $guard: expr) => {
        match $expression {
            $($pattern)|+ => $guard,
            _ => false
        }
    };
}

pub fn main() {
    let foo = Some("-12");
    assert!(matches!(foo, Some(bar) if
        matches!(bar.char_at(0), '+' | '-') &&
        matches!(bar.char_at(1), '0'...'9')
    ));
}
