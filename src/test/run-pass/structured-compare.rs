

enum foo { large, small, }

impl foo : cmp::Eq {
    pure fn eq(&&other: foo) -> bool {
        (self as uint) == (other as uint)
    }
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
