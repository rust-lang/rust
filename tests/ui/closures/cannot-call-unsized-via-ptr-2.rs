#![feature(unsized_fn_params)]

fn main() {
    // CoerceMany "LUB"
    let f = if true { |_a| {} } else { |_b| {} };
    //~^ ERROR the size for values of type `[u8]` cannot be known at compilation time
    //~| ERROR the size for values of type `[u8]` cannot be known at compilation time

    let slice: Box<[u8]> = Box::new([1; 8]);
    f(*slice);
}
