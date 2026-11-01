#[derive(Clone)]
struct P<T> {
    x: T,
    y: f64,
}

impl<T> P<T> {
    fn y(&self, y: f64) -> Self { P{y, .. self.clone() } }
    //~^ ERROR mismatched types [E0308]
}

fn main() {}
