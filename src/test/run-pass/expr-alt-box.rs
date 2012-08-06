


// -*- rust -*-

// Tests for match as expressions resulting in boxed types
fn test_box() {
    let res = match check true { true => { @100 } };
    assert (*res == 100);
}

fn test_str() {
    let res = match check true { true => { ~"happy" } };
    assert (res == ~"happy");
}

fn main() { test_box(); test_str(); }
