// check-pass

// Repeating a *constant* of non-Copy type (not just a constant expression) is already stable.

const EMPTY: Vec<i32> = Vec::new();

pub fn bar() -> [Vec<i32>; 2] {
    [EMPTY; 2]
}

fn main() {
    let x = bar();
}
