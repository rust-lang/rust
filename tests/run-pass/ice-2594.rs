// Copyright 2014-2018 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

#![allow(dead_code, unused_variables)]

/// Should not trigger an ICE in `SpanlessHash` / `consts::constant`
///
/// Issue: https://github.com/rust-lang/rust-clippy/issues/2594

fn spanless_hash_ice() {
    let txt = "something";
    let empty_header: [u8; 1] = [1; 1];

    match txt {
        "something" => {
            let mut headers = [empty_header; 1];
        },
        "" => (),
        _ => (),
    }
}

fn main() {}
