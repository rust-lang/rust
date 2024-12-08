//@ check-pass

pub struct X {
    pub a: i32,
    pub b: i32,
}

fn main() {
    const DX: X = X { a: 0, b: 0 };
    const _X1: X = X { a: 1, ..DX };
    let _x2 = X { a: 1, b: 2, ..DX };
    const _X3: X = X { a: 1, b: 2, ..DX };
}
