// run-pass




struct Point {x: isize, y: isize}

pub fn main() {
    let origin: Point = Point {x: 0, y: 0};
    let right: Point = Point {x: origin.x + 10,.. origin};
    let up: Point = Point {y: origin.y + 10,.. origin};
    assert_eq!(origin.x, 0);
    assert_eq!(origin.y, 0);
    assert_eq!(right.x, 10);
    assert_eq!(right.y, 0);
    assert_eq!(up.x, 0);
    assert_eq!(up.y, 10);
}
