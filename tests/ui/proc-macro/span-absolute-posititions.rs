//@ proc-macro: assert-span-pos.rs
// ignore-tidy-tab
extern crate assert_span_pos;

assert_span_pos::assert_span_pos!(5, 35);

// Test space indentation
    assert_span_pos::assert_span_pos!(8, 39);
// Test tab indentation
	assert_span_pos::assert_span_pos!(10, 36);

// Two tests to ensure the promise of the docs that the column is the number
// of UTF-8 bytes instead of some other number like number of code points.

// Test that multi byte UTF-8 characters indeed count as multiple bytes
/*🌈*/assert_span_pos::assert_span_pos!(16, 40);
// Test with a complete grapheme cluster
/*🏳️‍🌈*/assert_span_pos::assert_span_pos!(18, 43);

// Test that the macro actually emits an error on a mismatch:
assert_span_pos::assert_span_pos!(0, 35); //~ ERROR line/column mismatch: (0, 35) != (21, 35)
assert_span_pos::assert_span_pos!(22, 0); //~ ERROR line/column mismatch: (22, 0) != (22, 35)

fn main() {}
