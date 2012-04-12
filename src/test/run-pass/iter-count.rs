fn main() {
    assert [].count(22u) == 0u;
    assert [1u, 3u].count(22u) == 0u;
    assert [22u, 1u, 3u].count(22u) == 1u;
    assert [22u, 1u, 22u].count(22u) == 2u;
    assert none.count(22u) == 0u;
    assert some(1u).count(22u) == 0u;
    assert some(22u).count(22u) == 1u;
}