fn is_even(+x: uint) -> bool { (x % 2) == 0 }

fn main() {
    assert [1, 3].filter_to_vec(is_even) == ~[];
    assert [1, 2, 3].filter_to_vec(is_even) == ~[2];
    assert None.filter_to_vec(is_even) == ~[];
    assert Some(1).filter_to_vec(is_even) == ~[];
    assert Some(2).filter_to_vec(is_even) == ~[2];
}
