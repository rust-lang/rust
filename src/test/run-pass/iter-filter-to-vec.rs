fn is_even(&&x: uint) -> bool { (x % 2u) == 0u }

fn main() {
    assert [1u, 3u]/_.filter_to_vec(is_even) == ~[];
    assert [1u, 2u, 3u]/_.filter_to_vec(is_even) == ~[2u];
    assert None.filter_to_vec(is_even) == ~[];
    assert Some(1u).filter_to_vec(is_even) == ~[];
    assert Some(2u).filter_to_vec(is_even) == ~[2u];
}
