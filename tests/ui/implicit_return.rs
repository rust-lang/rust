// Copyright 2014-2018 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

#![warn(clippy::implicit_return)]

fn test_end_of_fn() -> bool {
    if true {
        // no error!
        return true;
    }
    true
}

#[allow(clippy::needless_bool)]
fn test_if_block() -> bool {
    if true {
        true
    } else {
        false
    }
}

#[allow(clippy::match_bool)]
fn test_match(x: bool) -> bool {
    match x {
        true => false,
        false => true,
    }
}

#[allow(clippy::never_loop)]
fn test_loop() -> bool {
    loop {
        break true;
    }
}

fn test_closure() {
    let _ = || true;
    let _ = || true;
}

fn main() {
    let _ = test_end_of_fn();
    let _ = test_if_block();
    let _ = test_match(true);
    let _ = test_loop();
    test_closure();
}
