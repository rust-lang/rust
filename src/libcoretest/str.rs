// Copyright 2014 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

fn check_contains_all_substrings(s: &str) {
    assert!(s.contains(""));
    for i in range(0, s.len()) {
        for j in range(i+1, s.len() + 1) {
            assert!(s.contains(s.slice(i, j)));
        }
    }
}

#[test]
fn strslice_issue_16589() {
    assert!("bananas".contains("nana"));

    // prior to the fix for #16589, x.contains("abcdabcd") returned false
    // test all substrings for good measure
    check_contains_all_substrings("012345678901234567890123456789bcdabcdabcd");
}


#[test]
fn test_strslice_contains() {
    let x = "There are moments, Jeeves, when one asks oneself, 'Do trousers matter?'";
    check_contains_all_substrings(x);
}
