


// -*- rust -*-

// Tests for alt as expressions resulting in boxed types
fn test_box() {
    auto res = alt (true) { case (true) { @100 } };
    assert (*res == 100);
}

fn test_str() {
    auto res = alt (true) { case (true) { "happy" } };
    assert (res == "happy");
}

fn main() { test_box(); test_str(); }