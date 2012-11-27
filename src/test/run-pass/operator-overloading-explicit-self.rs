struct S {
    x: int
}

impl S {
    pure fn add(&self, other: &S) -> S {
        S { x: self.x + other.x }
    }
}

fn main() {
    let mut s = S { x: 1 };
    s += S { x: 2 };
    assert s.x == 3;
}

