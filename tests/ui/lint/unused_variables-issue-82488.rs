//@ run-rustfix
#![deny(unused_variables)]

struct Point {
    x: u32,
    y: u32
}

fn process_point(Point { x, y: renamed }: Point) {
//~^ ERROR unused variable: `renamed`
    let _ = x;
}

fn main() {
    process_point(Point { x: 0, y: 0 });
}
