struct Point { x: isize, y: isize }
struct Empty;

fn main() {
    let origin = Point { x: 0, y: 0 };
    origin.0;
    //~^ ERROR no field `0` on type `Point`
    Empty.0;
    //~^ ERROR no field `0` on type `Empty`
}
