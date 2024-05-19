// Tests to make sure that parens are needed for method calls without arguments.
// outputs text to make sure either an anonymous function is provided or
// open-close '()' parens are given


struct Point {
    x: isize,
    y: isize
}
impl Point {
    fn new() -> Point {
        Point{x:0, y:0}
    }
    fn get_x(&self) -> isize {
        self.x
    }
}

fn main() {
    let point: Point = Point::new();
    let px: isize =  point
                        .get_x;//~ ERROR attempted to take value of method `get_x` on type `Point`

    // Ensure the span is useful
    let ys = &[1,2,3,4,5,6,7];
    let a = ys.iter()
              .map(|x| x)
              .filter(|&&x| x == 1)
              .filter_map; //~ ERROR attempted to take value of method `filter_map` on type
}
