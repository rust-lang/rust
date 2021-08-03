// aux-build:assert-span-pos.rs
// ignore-tidy-tab
extern crate assert_span_pos;

assert_span_pos::assert_span_pos!(5, 35);

// Test space indentation
    assert_span_pos::assert_span_pos!(8, 39);
// Test tab indentation
	assert_span_pos::assert_span_pos!(10, 36);

// Test that the macro actually emits an error on a mismatch:
assert_span_pos::assert_span_pos!(0, 35); //~ ERROR line/column mismatch: (0, 35) != (13, 35)
assert_span_pos::assert_span_pos!(14, 0); //~ ERROR line/column mismatch: (14, 0) != (14, 35)

fn main() {}
