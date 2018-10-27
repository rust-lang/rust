// run-pass

#![feature(unsized_locals)]

fn main() {
    let foo: Box<[u8]> = Box::new(*b"foo");
    let foo: [u8] = *foo;
    assert_eq!(&foo, b"foo" as &[u8]);
}
