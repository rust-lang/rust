//@ run-pass
#[derive(PartialEq, Debug)]
struct Point { x : isize }

pub fn main() {
    assert_eq!(14,14);
    assert_eq!("abc".to_string(),"abc".to_string());
    assert_eq!(Box::new(Point{x:34}),Box::new(Point{x:34}));
    assert_eq!(&Point{x:34},&Point{x:34});
    assert_eq!(42, 42, "foo bar");
    assert_eq!(42, 42, "a {} c", "b");
    assert_eq!(42, 42, "{x}, {y}, {z}", x = 1, y = 2, z = 3);
}
