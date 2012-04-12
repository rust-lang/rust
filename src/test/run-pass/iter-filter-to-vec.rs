fn is_even(&&x: uint) -> bool { (x % 2u) == 0u }

fn main() {
    assert [1u, 3u].filter_to_vec(is_even) == [];
    assert [1u, 2u, 3u].filter_to_vec(is_even) == [2u];
    assert none.filter_to_vec(is_even) == [];
    assert some(1u).filter_to_vec(is_even) == [];
    assert some(2u).filter_to_vec(is_even) == [2u];
}