struct Rectangle {
    width: i32,
    height: i32,
}
impl Rectangle {
    fn new(width: i32, height: i32) -> Self {
        Self { width, height }
    }
}

fn main() {
    let rect = Rectangle::new(3, 4);
    // `area` is not implemented for `Rectangle`, so this should not suggest
    let _ = rect::area();
    //~^ ERROR failed to resolve: use of unresolved module or unlinked crate `rect`
}
