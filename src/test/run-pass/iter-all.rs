fn is_even(&&x: uint) -> bool { (x % 2u) == 0u }

fn main() {
    assert ![1u, 2u]/_.all(is_even);
    assert [2u, 4u]/_.all(is_even);
    assert []/_.all(is_even);

    assert !Some(1u).all(is_even);
    assert Some(2u).all(is_even);
    assert None.all(is_even);
}
