// xfail-fast
#[legacy_modes];

struct Point {
    x: int,
    y: int
}

impl Point : ops::Add<Point,Point> {
    pure fn add(&self, other: &Point) -> Point {
        Point {x: self.x + (*other).x, y: self.y + (*other).y}
    }
}

impl Point : ops::Sub<Point,Point> {
    pure fn sub(&self, other: &Point) -> Point {
        Point {x: self.x - (*other).x, y: self.y - (*other).y}
    }
}

impl Point : ops::Neg<Point> {
    pure fn neg(&self) -> Point {
        Point {x: -self.x, y: -self.y}
    }
}

impl Point : ops::Index<bool,int> {
    pure fn index(&self, +x: bool) -> int {
        if x { self.x } else { self.y }
    }
}

impl Point : cmp::Eq {
    pure fn eq(&self, other: &Point) -> bool {
        (*self).x == (*other).x && (*self).y == (*other).y
    }
    pure fn ne(&self, other: &Point) -> bool { !(*self).eq(other) }
}

fn main() {
    let mut p = Point {x: 10, y: 20};
    p += Point {x: 101, y: 102};
    p = p - Point {x: 100, y: 100};
    assert p + Point {x: 5, y: 5} == Point {x: 16, y: 27};
    assert -p == Point {x: -11, y: -22};
    assert p[true] == 11;
    assert p[false] == 22;
    // Issue #1733
    fn~(_x: int){}(p[true]);
}
