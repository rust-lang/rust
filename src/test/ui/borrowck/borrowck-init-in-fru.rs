// revisions: ast mir
//[mir]compile-flags: -Z borrowck=mir

#[derive(Clone)]
struct Point {
    x: isize,
    y: isize,
}

fn main() {
    let mut origin: Point;
    origin = Point { x: 10, ..origin };
    //[ast]~^ ERROR use of possibly uninitialized variable: `origin.y` [E0381]
    //[mir]~^^ ERROR [E0381]
    origin.clone();
}
