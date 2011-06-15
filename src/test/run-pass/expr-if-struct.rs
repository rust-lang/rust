


// -*- rust -*-

// Tests for if as expressions returning structural types
fn test_rec() {
    auto res = if (true) { rec(i=100) } else { rec(i=101) };
    assert (res == rec(i=100));
}

fn test_tag() {
    tag mood { happy; sad; }
    auto res = if (true) { happy } else { sad };
    assert (res == happy);
}

fn main() { test_rec(); test_tag(); }