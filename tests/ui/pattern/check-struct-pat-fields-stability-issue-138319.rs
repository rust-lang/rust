//@ check-pass
struct Point {
    #[deprecated = "x is deprecated"]
    _x: i32,
    _y: i32,
}

fn main() {
    let p = Point { _x: 1, _y: 2 }; //~ WARNING use of deprecated field `Point::_x`
    // Before fix, it report an warning
    let Point { #[expect(deprecated)]_x, .. } = p;
}
