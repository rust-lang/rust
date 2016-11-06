// Copyright 2016 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

// Regression test for #18060: match arms were matching in the wrong order.

fn main() {
    assert_eq!(2, match (1, 3) { (0, 2...5) => 1, (1, 3) => 2, (_, 2...5) => 3, (_, _) => 4 });
    assert_eq!(2, match (1, 3) {                  (1, 3) => 2, (_, 2...5) => 3, (_, _) => 4 });
    assert_eq!(2, match (1, 7) { (0, 2...5) => 1, (1, 7) => 2, (_, 2...5) => 3, (_, _) => 4 });
}
