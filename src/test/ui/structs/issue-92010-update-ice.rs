// Regression test for issue #92010
// Tests that we don't ICE on an incorrect struct update usage

#[derive(Clone)]
struct P<T> {
    x: T,
    y: f64,
}

impl<T> P<T> {
    fn y(&self, y: f64) -> Self { P{y, .. self.clone() } } //~ ERROR mismatched types
}

fn main() {}
