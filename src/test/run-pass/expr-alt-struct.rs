


// -*- rust -*-

// Tests for alt as expressions resulting in structural types
fn test_rec() {
    auto rs = alt (true) { case (true) { rec(i=100) } };
    assert (rs == rec(i=100));
}

fn test_tag() {
    tag mood { happy; sad; }
    auto rs = alt (true) { case (true) { happy } case (false) { sad } };
    assert (rs == happy);
}

fn main() { test_rec(); test_tag(); }