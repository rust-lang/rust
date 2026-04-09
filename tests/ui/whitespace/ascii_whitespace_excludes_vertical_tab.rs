// This test checks that split_ascii_whitespace does not split on a
// vertical tab (\x0B). The standard library follows the WhatWG ASCII
// whitespace definition, which does not include vertical tab.
//
// So in this case, "a\x0Bb" should stay as a single piece.
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

    println!("All assertions passed.");
}