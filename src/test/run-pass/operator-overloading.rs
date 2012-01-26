type point = {x: int, y: int};

impl add_point for point {
    fn op_add(other: point) -> point {
        {x: self.x + other.x, y: self.y + other.y}
    }
    fn op_neg() -> point {
        {x: -self.x, y: -self.y}
    }
}

fn main() {
    let p = {x: 10, y: 20};
    p += {x: 1, y: 2};
    assert p + {x: 5, y: 5} == {x: 16, y: 27};
    assert -p == {x: -11, y: -22};
}
