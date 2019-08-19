// run-pass

fn main() {
    let v = vec![1, 2, 3];
    let boxed: Box<dyn Iterator<Item=i32>> = Box::new(v.into_iter());
    assert_eq!(boxed.max(), Some(3));

    let v = vec![1, 2, 3];
    let boxed: &mut dyn Iterator<Item=i32> = &mut v.into_iter();
    assert_eq!(boxed.max(), Some(3));
}
