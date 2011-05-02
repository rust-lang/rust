fn main() {
    auto v = vec(1);
    v += vec(2);
    v += vec(3);
    v += vec(4);
    v += vec(5);
    assert (v.(0) == 1);
    assert (v.(1) == 2);
    assert (v.(2) == 3);
    assert (v.(3) == 4);
    assert (v.(4) == 5);
}

