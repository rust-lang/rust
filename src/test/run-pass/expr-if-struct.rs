


// -*- rust -*-

// Tests for if as expressions returning structural types
fn test_rec() {
    let rs = if true { {i: 100} } else { {i: 101} };
    assert (rs.i == 100);
}

enum mood { happy, sad, }

impl mood : cmp::Eq {
    pure fn eq(&self, other: &mood) -> bool {
        ((*self) as uint) == ((*other) as uint)
    }
    pure fn ne(&self, other: &mood) -> bool { !(*self).eq(other) }
}

fn test_tag() {
    let rs = if true { happy } else { sad };
    assert (rs == happy);
}

fn main() { test_rec(); test_tag(); }
