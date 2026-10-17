//@ edition: 2018

// Do not suggest changing a function when the future is not its return value or when doing so
// would make a trait method incompatible with its declaration.

async fn number() -> i32 {
    42
}

// A local block tail does not contribute to the function's return value.
fn local_block() {
    let _: i32 = { number() };
    //~^ ERROR mismatched types
}

struct Wrapper;

trait Trait {
    fn trait_method() -> i32;
}

impl Trait for Wrapper {
    fn trait_method() -> i32 {
        number()
        //~^ ERROR mismatched types
    }
}

fn main() {}
