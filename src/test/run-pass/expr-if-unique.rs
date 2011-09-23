


// -*- rust -*-

// Tests for if as expressions returning boxed types
fn test_box() {
    let rs = if true { ~100 } else { ~101 };
    assert (*rs == 100);
}

fn main() { test_box(); }
