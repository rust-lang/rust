// Copyright 2014 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

// Test binary_search_by_key lifetime. Issue #34683

#[derive(Debug)]
struct Assignment {
    topic: String,
    partition: i32,
}

fn main() {
    let xs = vec![
        Assignment { topic: "abc".into(), partition: 1 },
        Assignment { topic: "def".into(), partition: 2 },
        Assignment { topic: "ghi".into(), partition: 3 },
    ];

    let key: &str = "def";
    let r = xs.binary_search_by_key(&key, |e| &e.topic);
    assert_eq!(Ok(1), r.map(|i| i));
}
