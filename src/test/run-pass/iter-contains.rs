fn main() {
    assert [].contains(&22u) == false;
    assert [1u, 3u].contains(&22u) == false;
    assert [22u, 1u, 3u].contains(&22u) == true;
    assert [1u, 22u, 3u].contains(&22u) == true;
    assert [1u, 3u, 22u].contains(&22u) == true;
    assert None.contains(&22u) == false;
    assert Some(1u).contains(&22u) == false;
    assert Some(22u).contains(&22u) == true;
}
