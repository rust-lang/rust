fn main() {
    assert [1u, 3u]/_.to_vec() == ~[1u, 3u];
    let e: ~[uint] = ~[];
    assert e.to_vec() == ~[];
    assert None::<uint>.to_vec() == ~[];
    assert Some(1u).to_vec() == ~[1u];
    assert Some(2u).to_vec() == ~[2u];
}
