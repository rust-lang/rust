// Copyright 2014 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

// error-pattern: too big for the current architecture

#[cfg(target_word_size = "64")]
fn main() {
    let n = 0u;
    let a = box [&n,..0xF000000000000000u];
    println!("{}", a[0xFFFFFFu]);
}

#[cfg(target_word_size = "32")]
fn main() {
    let n = 0u;
    let a = box [&n,..0xFFFFFFFFu];
    println!("{}", a[0xFFFFFFu]);
}
