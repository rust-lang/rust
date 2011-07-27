


// -*- rust -*-

// Tests for standalone blocks as expressions
fn test_basic() { let rs: bool = { true }; assert (rs); }

fn test_rec() { let rs = { {v1: 10, v2: 20} }; assert (rs.v2 == 20); }

fn test_filled_with_stuff() {
    let rs = { let a = 0; while a < 10 { a += 1; } a };
    assert (rs == 10);
}

fn main() { test_basic(); test_rec(); test_filled_with_stuff(); }