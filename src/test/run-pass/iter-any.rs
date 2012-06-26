fn is_even(&&x: uint) -> bool { (x % 2u) == 0u }

fn main() {
    assert ![1u, 3u]/~.any(is_even);
    assert [1u, 2u]/~.any(is_even);
    assert ![]/~.any(is_even);

    assert !some(1u).any(is_even);
    assert some(2u).any(is_even);
    assert !none.any(is_even);
}