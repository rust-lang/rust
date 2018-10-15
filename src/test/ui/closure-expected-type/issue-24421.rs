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

fn test<F: Fn(&u64, &u64)>(f: F) {}

fn main() {
    test(|x,      y     | {});
    test(|x:&u64, y:&u64| {});
    test(|x:&u64, y     | {});
    test(|x,      y:&u64| {});
}
