mod foo {
    pub struct Point {
        pub x: i32,
        pub y: i32,
    }
}

impl foo::Point {
    fn x(&self) -> i32 { self.x }
}

fn main() {
    assert_eq!((foo::Point { x: 1, y: 3}).x(), 1);
}
