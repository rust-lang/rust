// check-pass

trait Zero {
    const ZERO: Self;
}

impl Zero for i32 {
    const ZERO: Self = 0;
}

fn main() {
    match 1 {
        Zero::ZERO ..= 1 => {},
        _ => {},
    }
}
