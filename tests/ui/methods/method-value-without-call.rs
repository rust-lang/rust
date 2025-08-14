//! Test taking a method value without parentheses

struct Point {
    x: isize,
    y: isize,
}

impl Point {
    fn new() -> Point {
        Point { x: 0, y: 0 }
    }

    fn get_x(&self) -> isize {
        self.x
    }
}

fn main() {
    // Test with primitive type method
    let _f = 10i32.abs; //~ ERROR attempted to take value of method

    // Test with custom type method
    let point: Point = Point::new();
    let px: isize = point.get_x; //~ ERROR attempted to take value of method `get_x` on type `Point`

    // Test with method chains - ensure the span is useful
    let ys = &[1, 2, 3, 4, 5, 6, 7];
    let a = ys
        .iter()
        .map(|x| x)
        .filter(|&&x| x == 1)
        .filter_map; //~ ERROR attempted to take value of method `filter_map` on type
}
