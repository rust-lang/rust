fn test(foo: Box<Vec<isize>>) {
    assert_eq!((*foo)[0], 10);
}

pub fn main() {
    let x = Box::new(vec![10]);
    // Test forgetting a local by move-in
    test(x);
}
