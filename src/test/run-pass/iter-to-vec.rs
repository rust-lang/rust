fn main() {
    assert [1u, 3u]/_.to_vec() == ~[1u, 3u];
    let e: ~[uint] = ~[];
    assert e.to_vec() == ~[];
    assert none::<uint>.to_vec() == ~[];
    assert some(1u).to_vec() == ~[1u];
    assert some(2u).to_vec() == ~[2u];
}
