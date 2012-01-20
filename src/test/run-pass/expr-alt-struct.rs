


// -*- rust -*-

// Tests for alt as expressions resulting in structural types
fn test_rec() {
    let rs = alt true { true { {i: 100} } };
    assert (rs == {i: 100});
}

fn test_tag() {
    enum mood { happy; sad; }
    let rs = alt true { true { happy } false { sad } };
    assert (rs == happy);
}

fn main() { test_rec(); test_tag(); }
