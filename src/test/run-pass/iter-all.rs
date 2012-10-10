fn is_even(x: &uint) -> bool { (*x % 2) == 0 }

fn main() {
    assert ![1u, 2u].all(is_even);
    assert [2u, 4u].all(is_even);
    assert [].all(is_even);

    assert !Some(1u).all(is_even);
    assert Some(2u).all(is_even);
    assert None.all(is_even);
}
