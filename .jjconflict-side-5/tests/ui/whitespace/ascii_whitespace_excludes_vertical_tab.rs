//@ run-pass
// This test checks that split_ascii_whitespace does NOT split on
// vertical tab (\x0B), because the standard library uses the WhatWG
// Infra Standard definition of ASCII whitespace, which excludes
// vertical tab.
//
// See: https://github.com/rust-lang/rust-project-goals/issues/53

fn main() {
    let s = "a\x0Bb";

    let parts: Vec<&str> = s.split_ascii_whitespace().collect();

    assert_eq!(parts.len(), 1,
        "vertical tab should not be treated as ASCII whitespace");

    let s2 = "a b";
    let parts2: Vec<&str> = s2.split_ascii_whitespace().collect();
    assert_eq!(parts2.len(), 2,
        "regular space should split correctly");

}
