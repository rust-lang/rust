


// -*- rust -*-

// Tests for alt as expressions resulting in boxed types
fn test_box() { let res = alt true { true { @100 } }; assert (*res == 100); }

fn test_str() {
    let res = alt true { true { "happy" } };
    assert (res == "happy");
}

fn main() { test_box(); test_str(); }
