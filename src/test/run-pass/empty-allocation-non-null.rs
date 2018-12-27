pub fn main() {
    assert!(Some(Box::new(())).is_some());

    let xs: Box<[()]> = Box::<[(); 0]>::new([]);
    assert!(Some(xs).is_some());

    struct Foo;
    assert!(Some(Box::new(Foo)).is_some());

    let ys: Box<[Foo]> = Box::<[Foo; 0]>::new([]);
    assert!(Some(ys).is_some());
}
