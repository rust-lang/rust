struct Point {
    x: f64,
    y: f64,
}

trait ToString_ {
    fn to_string(&self) -> String;
}

impl ToString_ for Point {
    fn new(x: f64, y: f64) -> Point {
    //~^ ERROR method `new` is not a member of trait `ToString_`
        Point { x: x, y: y }
    }

    fn to_string(&self) -> String {
        format!("({}, {})", self.x, self.y)
    }
}

fn main() {
    let p = Point::new(0.0, 0.0);
    //~^ ERROR no function or associated item named `new` found for struct `Point`
    println!("{}", p.to_string());
}
