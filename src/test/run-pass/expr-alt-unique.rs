


// -*- rust -*-

// Tests for match as expressions resulting in boxed types
fn test_box() {
    let res = match check true { true => { ~100 } };
    assert (*res == 100);
}

fn main() { test_box(); }
