trait sum {
    fn sum() -> int;
}

// Note: impl on a slice
impl &[int]: sum {
    fn sum() -> int {
        let mut sum = 0;
        for vec::each(self) |e| { sum += e; }
        return sum;
    }
}

fn call_sum(x: &[int]) -> int { x.sum() }

fn main() {
    let x = ~[1, 2, 3];
    let y = call_sum(x);
    debug!("y==%d", y);
    assert y == 6;

    let x = ~[mut 1, 2, 3];
    let y = x.sum();
    debug!("y==%d", y);
    assert y == 6;

    let x = ~[1, 2, 3];
    let y = x.sum();
    debug!("y==%d", y);
    assert y == 6;
}
