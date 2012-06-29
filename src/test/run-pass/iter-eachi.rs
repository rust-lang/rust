fn main() {
    let mut c = 0u;
    for [1u, 2u, 3u, 4u, 5u]/_.eachi { |i, v|
        assert (i + 1u) == v;
        c += 1u;
    }
    assert c == 5u;

    for none::<uint>.eachi { |i, v| fail; }

    let mut c = 0u;
    for some(1u).eachi { |i, v|
        assert (i + 1u) == v;
        c += 1u;
    }
    assert c == 1u;

}
