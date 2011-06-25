


// -*- rust -*-

// Tests for standalone blocks as expressions
fn test_basic() { let bool rs = { true }; assert (rs); }

fn test_rec() { auto rs = { rec(v1=10, v2=20) }; assert (rs.v2 == 20); }

fn test_filled_with_stuff() {
    auto rs = { auto a = 0; while (a < 10) { a += 1; } a };
    assert (rs == 10);
}

fn main() { test_basic(); test_rec(); test_filled_with_stuff(); }