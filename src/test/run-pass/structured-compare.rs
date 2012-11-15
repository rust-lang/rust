

enum foo { large, small, }

impl foo : cmp::Eq {
    pure fn eq(&self, other: &foo) -> bool {
        ((*self) as uint) == ((*other) as uint)
    }
    pure fn ne(&self, other: &foo) -> bool { !(*self).eq(other) }
}

fn main() {
    let a = (1, 2, 3);
    let b = (1, 2, 3);
    assert (a == b);
    assert (a != (1, 2, 4));
    assert (a < (1, 2, 4));
    assert (a <= (1, 2, 4));
    assert ((1, 2, 4) > a);
    assert ((1, 2, 4) >= a);
    let x = large;
    let y = small;
    assert (x != y);
    assert (x == large);
    assert (x != small);
}
