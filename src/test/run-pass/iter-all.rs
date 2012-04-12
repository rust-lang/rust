fn is_even(&&x: uint) -> bool { (x % 2u) == 0u }

fn main() {
    assert ![1u, 2u].all(is_even);
    assert [2u, 4u].all(is_even);
    assert [].all(is_even);

    assert !some(1u).all(is_even);
    assert some(2u).all(is_even);
    assert none.all(is_even);
}