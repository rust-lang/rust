fn takes_two(x: &int, y: &int) -> int { *x + *y }

fn has_two(x: &a/int, y: &b/int) -> int {
    takes_two(x, y)
}

fn main() {
    assert has_two(&20, &2) == 22;
}