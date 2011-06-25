


// -*- rust -*-

// Tests for alt as expressions resulting in boxed types
fn test_box() {
    auto rs = alt (true) { case (true) { @100 } };
    assert (*rs == 100);
}

fn test_str() {
    auto rs = alt (true) { case (true) { "happy" } };
    assert (rs == "happy");
}

fn main() { test_box(); test_str(); }