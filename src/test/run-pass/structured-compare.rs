

tag foo { large; small; }

fn main() {
    auto a = tup(1, 2, 3);
    auto b = tup(1, 2, 3);
    assert (a == b);
    assert (a != tup(1, 2, 4));
    assert (a < tup(1, 2, 4));
    assert (a <= tup(1, 2, 4));
    assert (tup(1, 2, 4) > a);
    assert (tup(1, 2, 4) >= a);
    auto x = large;
    auto y = small;
    assert (x != y);
    assert (x == large);
    assert (x != small);
}