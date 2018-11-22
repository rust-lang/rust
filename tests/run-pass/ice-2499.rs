// Copyright 2014-2018 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.




#![allow(dead_code, clippy::char_lit_as_u8, clippy::needless_bool)]

/// Should not trigger an ICE in `SpanlessHash` / `consts::constant`
///
/// Issue: https://github.com/rust-lang/rust-clippy/issues/2499

fn f(s: &[u8]) -> bool {
    let t = s[0] as char;

    match t {
        'E' | 'W' => {}
        'T' => if s[0..4] != ['0' as u8; 4] {
            return false;
        } else {
            return true;
        },
        _ => {
            return false;
        }
    }
    true
}

fn main() {}
