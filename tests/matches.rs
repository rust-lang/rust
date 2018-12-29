// Copyright 2014-2018 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

#![feature(rustc_private)]

extern crate syntax;
use std::collections::Bound;

#[test]
fn test_overlapping() {
    use crate::syntax::source_map::DUMMY_SP;
    use clippy_lints::matches::overlapping;

    let sp = |s, e| clippy_lints::matches::SpannedRange {
        span: DUMMY_SP,
        node: (s, e),
    };

    assert_eq!(None, overlapping::<u8>(&[]));
    assert_eq!(None, overlapping(&[sp(1, Bound::Included(4))]));
    assert_eq!(
        None,
        overlapping(&[sp(1, Bound::Included(4)), sp(5, Bound::Included(6))])
    );
    assert_eq!(
        None,
        overlapping(&[
            sp(1, Bound::Included(4)),
            sp(5, Bound::Included(6)),
            sp(10, Bound::Included(11))
        ],)
    );
    assert_eq!(
        Some((&sp(1, Bound::Included(4)), &sp(3, Bound::Included(6)))),
        overlapping(&[sp(1, Bound::Included(4)), sp(3, Bound::Included(6))])
    );
    assert_eq!(
        Some((&sp(5, Bound::Included(6)), &sp(6, Bound::Included(11)))),
        overlapping(&[
            sp(1, Bound::Included(4)),
            sp(5, Bound::Included(6)),
            sp(6, Bound::Included(11))
        ],)
    );
}
