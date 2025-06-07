fn main() {
    let foo: Box<[u8]> = Box::new(*b"foo");
    let foo: [u8] = *foo; //~ERROR the size for values of type `[u8]` cannot be known at compilation time [E0277]
    assert_eq!(&foo, b"foo" as &[u8]);
}
