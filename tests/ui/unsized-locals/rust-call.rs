#![feature(unsized_tuple_coercion)]
#![feature(unboxed_closures)]
#![feature(unsized_fn_params)]

fn bad() -> extern "rust-call" fn(([u8],)) { todo!() }

fn main() {
    let f = bad();
    let slice: Box<([u8],)> = Box::new(([1; 8],));
    f(*slice);
    //~^ ERROR the size for values of type `[u8]` cannot be known at compilation time
}
