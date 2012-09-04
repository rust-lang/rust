fn is_even(&&x: uint) -> bool { (x % 2u) == 0u }

fn main() {
    assert ![1u, 3u]/_.any(is_even);
    assert [1u, 2u]/_.any(is_even);
    assert ![]/_.any(is_even);

    assert !Some(1u).any(is_even);
    assert Some(2u).any(is_even);
    assert !None.any(is_even);
}
