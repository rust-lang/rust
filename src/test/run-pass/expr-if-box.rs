


// -*- rust -*-

// Tests for if as expressions returning boxed types
fn test_box() {
    let rs = if true { @100 } else { @101 };
    assert (*rs == 100);
}

fn test_str() {
    let rs = if true { "happy" } else { "sad" };
    assert (rs == "happy");
}

fn main() { test_box(); test_str(); }
