// Issue #3656
// Issue Name: pub method preceeded by attribute can't be parsed
// Abstract: Visibility parsing failed when compiler parsing

struct Point {
    x: float,
    y: float
}

pub enum Shape {
    Circle(Point, float),
    Rectangle(Point, Point)
}

pub impl Shape {
    pub fn area(sh: Shape) -> float {
        match sh {
            Circle(_, size) => float::consts::pi * size * size,
            Rectangle(Point {x, y}, Point {x: x2, y: y2}) => (x2 - x) * (y2 - y)
        }
    }
}

fn main(){
    let s = Circle(Point { x: 1f, y: 2f }, 3f);
    io::println(fmt!("%f", s.area(s)));
}