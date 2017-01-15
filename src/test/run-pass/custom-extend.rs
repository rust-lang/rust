// Copyright 2016 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

// Ensure the blanket of FromIterator on String doesn't conflict with custom
// implementations (even if specialization changes).
// See: https://github.com/rust-lang/rust/pull/34389#issuecomment-227305533

use std::iter::FromIterator;

#[derive(Copy, Clone)]
struct S;

impl FromIterator<S> for String {
    fn from_iter<T: IntoIterator<Item=S>>(iter: T) -> String {
        let mut fin = String::new();
        for _ in iter {
            fin.push('a');
        }
        fin
    }
}

fn main() {
    let s: String = std::iter::repeat(S).take(5).collect();
    assert_eq!("aaaaa", s);
}
