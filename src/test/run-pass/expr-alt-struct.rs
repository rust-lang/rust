


// -*- rust -*-

// Tests for match as expressions resulting in structural types
fn test_rec() {
    let rs = match true { true => {i: 100}, _ => fail };
    assert (rs.i == 100);
}

enum mood { happy, sad, }

impl mood : cmp::Eq {
    pure fn eq(other: &mood) -> bool {
        (self as uint) == ((*other) as uint)
    }
    pure fn ne(other: &mood) -> bool { !self.eq(other) }
}

fn test_tag() {
    let rs = match true { true => { happy } false => { sad } };
    assert (rs == happy);
}

fn main() { test_rec(); test_tag(); }
