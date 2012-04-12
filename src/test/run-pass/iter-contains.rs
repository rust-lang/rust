fn main() {
    assert [].contains(22u) == false;
    assert [1u, 3u].contains(22u) == false;
    assert [22u, 1u, 3u].contains(22u) == true;
    assert [1u, 22u, 3u].contains(22u) == true;
    assert [1u, 3u, 22u].contains(22u) == true;
    assert none.contains(22u) == false;
    assert some(1u).contains(22u) == false;
    assert some(22u).contains(22u) == true;
}