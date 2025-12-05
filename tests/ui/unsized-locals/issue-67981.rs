#![feature(unsized_fn_params)]

fn main() {
    let f: fn([u8]) = |_| {};
    //~^ERROR the size for values of type `[u8]` cannot be known at compilation time
    let slice: Box<[u8]> = Box::new([1; 8]);

    f(*slice);
}
