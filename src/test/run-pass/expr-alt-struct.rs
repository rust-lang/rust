


// -*- rust -*-

// Tests for match as expressions resulting in structural types
fn test_rec() {
    let rs = match check true { true => { {i: 100} } };
    assert (rs == {i: 100});
}

fn test_tag() {
    enum mood { happy, sad, }
    let rs = match true { true => { happy } false => { sad } };
    assert (rs == happy);
}

fn main() { test_rec(); test_tag(); }
