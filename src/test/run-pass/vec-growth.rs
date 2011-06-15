

fn main() {
    auto v = [1];
    v += [2];
    v += [3];
    v += [4];
    v += [5];
    assert (v.(0) == 1);
    assert (v.(1) == 2);
    assert (v.(2) == 3);
    assert (v.(3) == 4);
    assert (v.(4) == 5);
}