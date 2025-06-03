fn main() {
    let foo: Box<[u8]> = Box::new(*b"foo");
    let _foo: [u8] = *foo; //~ERROR the size for values of type `[u8]` cannot be known at compilation time [E0277]
}
