type point = {x: int, y: int};

impl point_ops for point {
    fn +(other: point) -> point {
        {x: self.x + other.x, y: self.y + other.y}
    }
    fn -(other: point) -> point {
        {x: self.x - other.x, y: self.y - other.y}
    }
    fn unary-() -> point {
        {x: -self.x, y: -self.y}
    }
    fn [](x: bool) -> int {
        if x { self.x } else { self.y }
    }
}

fn main() {
    let p = {x: 10, y: 20};
    p += {x: 101, y: 102};
    p -= {x: 100, y: 100};
    assert p + {x: 5, y: 5} == {x: 16, y: 27};
    assert -p == {x: -11, y: -22};
    assert p[true] == 11;
    assert p[false] == 22;
    // Issue #1733
    fn~(_x: int){}(p[true]);
}
