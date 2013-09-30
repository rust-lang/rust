// Copyright 2013 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

fn test1() {
    // from issue 6338
    match ((1, ~"a"), (2, ~"b")) {
        ((1, a), (2, b)) | ((2, b), (1, a)) => {
                assert_eq!(a, ~"a");
                assert_eq!(b, ~"b");
            },
            _ => fail2!(),
    }
}

fn test2() {
    match (1, 2, 3) {
        (1, a, b) | (2, b, a) => {
            assert_eq!(a, 2);
            assert_eq!(b, 3);
        },
        _ => fail2!(),
    }
}

fn test3() {
    match (1, 2, 3) {
        (1, ref a, ref b) | (2, ref b, ref a) => {
            assert_eq!(*a, 2);
            assert_eq!(*b, 3);
        },
        _ => fail2!(),
    }
}

fn test4() {
    match (1, 2, 3) {
        (1, a, b) | (2, b, a) if a == 2 => {
            assert_eq!(a, 2);
            assert_eq!(b, 3);
        },
        _ => fail2!(),
    }
}

fn test5() {
    match (1, 2, 3) {
        (1, ref a, ref b) | (2, ref b, ref a) if *a == 2 => {
            assert_eq!(*a, 2);
            assert_eq!(*b, 3);
        },
        _ => fail2!(),
    }
}

pub fn main() {
    test1();
    test2();
    test3();
    test4();
    test5();
}
