//@ run-pass
#[derive(PartialEq, Debug)]
struct Point { x : isize }

pub fn main() {
    assert_ne!(666,14);
    assert_ne!("666".to_string(),"abc".to_string());
    assert_ne!(Box::new(Point{x:666}),Box::new(Point{x:34}));
    assert_ne!(&Point{x:666},&Point{x:34});
    assert_ne!(666, 42, "no gods no masters");
    assert_ne!(666, 42, "6 {} 6", "6");
    assert_ne!(666, 42, "{x}, {y}, {z}", x = 6, y = 6, z = 6);
}
