


// -*- rust -*-

// Tests for match as expressions resulting in boxed types
fn test_box() {
    let res = match true { true => { @100 } _ => fail ~"wat" };
    assert (*res == 100);
}

fn test_str() {
    let res = match true { true => { ~"happy" },
                         _ => fail ~"not happy at all" };
    assert (res == ~"happy");
}

fn main() { test_box(); test_str(); }
