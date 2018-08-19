// Copyright 2018 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

// This test illustrates that under NLL, we can remove our overly
// conservative approach for disallowing mutations of match inputs.

// See further discussion on rust-lang/rust#24535 and
// rust-lang/rfcs#1006.

// compile-flags: -Z disable-ast-check-for-mutation-in-guard

#![feature(nll)]

fn main() {
    rust_issue_24535();
    rfcs_issue_1006_1();
    rfcs_issue_1006_2();
}

fn rust_issue_24535() {
    fn compare(a: &u8, b: &mut u8) -> bool {
        a == b
    }

    let a = 3u8;

    match a {
        0 => panic!("nope"),
        3 if compare(&a, &mut 3) => (),
        _ => panic!("nope"),
    }
}

fn rfcs_issue_1006_1() {
    let v = vec!["1".to_string(), "2".to_string(), "3".to_string()];
    match Some(&v) {
        Some(iv) if iv.iter().any(|x| &x[..]=="2") => true,
        _ => panic!("nope"),
    };
}

fn rfcs_issue_1006_2() {
    #[inline(always)]
    fn check<'a, I: Iterator<Item=&'a i32>>(mut i: I) -> bool {
        i.any(|&x| x == 2)
    }

    let slice = [1, 2, 3];

    match 42 {
        _ if slice.iter().any(|&x| x == 2) => { true },
        _ => { panic!("nope"); }
    };

    // (This match is just illustrating how easy it was to circumvent
    // the checking performed for the previous `match`.)
    match 42 {
        _ if check(slice.iter()) => { true },
        _ => { panic!("nope"); }
    };
}
