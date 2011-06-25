


// -*- rust -*-

// Tests for if as expressions returning structural types
fn test_rec() {
    auto rs = if (true) { rec(i=100) } else { rec(i=101) };
    assert (rs == rec(i=100));
}

fn test_tag() {
    tag mood { happy; sad; }
    auto rs = if (true) { happy } else { sad };
    assert (rs == happy);
}

fn main() { test_rec(); test_tag(); }