#![feature(unsized_locals)]

fn main() {
    struct A<X: ?Sized>(X);
    A as fn(str) -> A<str>;
    //~^ERROR the size for values of type `[u8]` cannot be known at compilation time
}
