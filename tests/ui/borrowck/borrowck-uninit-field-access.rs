// Check that do not allow access to fields of uninitialized or moved
// structs.

#[derive(Default)]
struct Point {
    x: isize,
    y: isize,
}

#[derive(Default)]
struct Line {
    origin: Point,
    middle: Point,
    target: Point,
}

impl Line { fn consume(self) { } }

fn main() {
    let mut a: Point;
    let _ = a.x + 1; //~ ERROR [E0381]

    let mut line1 = Line::default();
    let _moved = line1.origin;
    let _ = line1.origin.x + 1; //~ ERROR [E0382]

    let mut line2 = Line::default();
    let _moved = (line2.origin, line2.middle);
    line2.consume(); //~ ERROR [E0382]
}
