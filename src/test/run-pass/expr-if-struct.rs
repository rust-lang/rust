


// -*- rust -*-

// Tests for if as expressions returning structural types
fn test_rec() {
    let rs = if true { {i: 100} } else { {i: 101} };
    assert (rs == {i: 100});
}

fn test_tag() {
    enum mood { happy; sad; }
    let rs = if true { happy } else { sad };
    assert (rs == happy);
}

fn main() { test_rec(); test_tag(); }
