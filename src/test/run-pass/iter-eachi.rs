fn main() {
    let mut c = 0u;
    for [1u, 2u, 3u, 4u, 5u].eachi |i, v| {
        assert (i + 1u) == *v;
        c += 1u;
    }
    assert c == 5u;

    for None::<uint>.eachi |i, v| { fail; }

    let mut c = 0u;
    for Some(1u).eachi |i, v| {
        assert (i + 1u) == *v;
        c += 1u;
    }
    assert c == 1u;

}
