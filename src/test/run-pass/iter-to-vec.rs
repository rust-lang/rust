fn main() {
    assert [1u, 3u].to_vec() == [1u, 3u];
    assert [].to_vec() == [];
    assert none.to_vec() == [];
    assert some(1u).to_vec() == [1u];
    assert some(2u).to_vec() == [2u];
}