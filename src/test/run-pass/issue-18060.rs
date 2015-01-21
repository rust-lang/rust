// Copyright 2015 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

fn main() {
    // Consider the second column of pattern matrix, if two range
    // patterns `2...5` are deemed equal, they'll end up in the same
    // subtree in pattern matrix reduction and crucially, before
    // the second branch. So we will get the wrong answer 3.
    let a = match (1, 3) {
        (0, 2...5) => 1,
        (1, 3) => 2,
        (_, 2...5) => 3,
        _ => 4us
    };
    assert_eq!(a, 2);

    let b = match (1, 3) {
        (0, 2...5) => 1,
        (1, 3) => 2,
        (_, 2...5) => 3,
        (_, _) => 4us
    };
    assert_eq!(b, 2);

    let c = match (1, 3) {
        (0, 2...5) => 1,
        (1, 3) => 2,
        (_, 3) => 3,
        (_, _) => 4us
    };
    assert_eq!(c, 2);

    // ditto, the same error will happen if two literal patterns `3`
    // are deemed equal.
    let d = match (1, 3) {
        (0, 3) => 1,
        (1, 2...5) => 2,
        (_, 3) => 3,
        (_, _) => 4us
    };
    assert_eq!(d, 2);

    let e = match (1, 3) {
        (0, 3) => 1,
        (2, 2...5) => 2,
        (_, 3) => 3,
        (_, _) => 4us
    };
    assert_eq!(e, 3);

    let f = match (2, 10) {
        (_, 9) => 1,
        (1, 10) => 2,
        (2, 10) => 3,
        (3, 1...9) => 4,
        (_, _) => 100
    };
    assert_eq!(f, 3);

    // OTOH, if certain column of pattern matrix consists only of literal
    // patterns, an LLVM switch instruction will be generated for such
    // column. If we don't group those literal patterns by their value,
    // some branches will be lost!
    let g = match (1, 3) {
        (0, 3) => 0,
        (1, 5) => 1,
        (1, 3) => 2,
        (_, _) => 4
    };
    assert_eq!(g, 2);
}
