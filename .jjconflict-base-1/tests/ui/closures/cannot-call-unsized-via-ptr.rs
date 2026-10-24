#![feature(unsized_fn_params)]

fn main() {
    // Simple coercion
    let f: fn([u8]) = |_result| {};
    //~^ ERROR the size for values of type `[u8]` cannot be known at compilation time

    let slice: Box<[u8]> = Box::new([1; 8]);
    f(*slice);
}
