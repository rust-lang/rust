struct Point {
    x: int;
    y: int;
}

impl Point : ops::add<Point,Point> {
    pure fn add(other: Point) -> Point {
        Point {x: self.x + other.x, y: self.y + other.y}
    }
}

impl Point : ops::sub<Point,Point> {
    pure fn sub(other: Point) -> Point {
        Point {x: self.x - other.x, y: self.y - other.y}
    }
}

impl Point : ops::neg<Point> {
    pure fn neg() -> Point {
        Point {x: -self.x, y: -self.y}
    }
}

impl Point : ops::index<bool,int> {
    pure fn index(&&x: bool) -> int {
        if x { self.x } else { self.y }
    }
}

fn main() {
    let mut p = Point {x: 10, y: 20};
    p += Point {x: 101, y: 102};
    p -= Point {x: 100, y: 100};
    assert p + Point {x: 5, y: 5} == Point {x: 16, y: 27};
    assert -p == Point {x: -11, y: -22};
    assert p[true] == 11;
    assert p[false] == 22;
    // Issue #1733
    fn~(_x: int){}(p[true]);
}
