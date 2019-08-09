#[derive(Clone)]
struct Point {
    x: isize,
    y: isize,
}

fn main() {
    let mut origin: Point;
    origin = Point { x: 10, ..origin };
    //~^ ERROR use of possibly uninitialized variable: `origin` [E0381]
    origin.clone();
}
