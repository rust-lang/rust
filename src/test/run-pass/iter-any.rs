fn is_even(x: &uint) -> bool { (*x % 2) == 0 }

fn main() {
    assert ![1u, 3u].any(is_even);
    assert [1u, 2u].any(is_even);
    assert ![].any(is_even);

    assert !Some(1).any(is_even);
    assert Some(2).any(is_even);
    assert !None.any(is_even);
}
