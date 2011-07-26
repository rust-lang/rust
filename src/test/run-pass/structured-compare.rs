

tag foo { large; small; }

fn main() {
    auto a = rec(x=1, y=2, z=3);
    auto b = rec(x=1, y=2, z=3);
    assert (a == b);
    assert (a != rec(x=1, y=2, z=4));
    assert (a < rec(x=1, y=2, z=4));
    assert (a <= rec(x=1, y=2, z=4));
    assert (rec(x=1, y=2, z=4) > a);
    assert (rec(x=1, y=2, z=4) >= a);
    auto x = large;
    auto y = small;
    assert (x != y);
    assert (x == large);
    assert (x != small);
}