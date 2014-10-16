// Copyright 2014 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

// error-pattern: ..1518599999

fn generic<T: Copy>(t: T) {
    let s: [T, ..1518600000] = [t, ..1518600000];
}

fn main() {
    let x: [u8, ..1518599999] = [0, ..1518599999];
    generic::<[u8, ..1518599999]>(x);
}
