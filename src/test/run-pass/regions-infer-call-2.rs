fn takes_two(x: &int, y: &int) -> int { *x + *y }

fn with<T>(f: fn(x: &int) -> T) -> T {
    f(&20)
}

fn has_one(x: &a/int) -> int {
    do with |y| { takes_two(x, y) }
}

fn main() {
    assert has_one(&2) == 22;
}