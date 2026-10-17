//@ edition: 2021
//@ run-rustfix

#![allow(dead_code, unused_must_use)]

// Suggest making eligible enclosing functions async when a return expression produces a future.

async fn number() -> i32 {
    42
}

async fn unit() {}

fn wrapped_number() -> i32 {
    number()
    //~^ ERROR mismatched types
}

fn explicit_return() -> i32 {
    return number();
    //~^ ERROR mismatched types
}

struct Wrapper;

impl Wrapper {
    pub unsafe fn inherent_method() -> i32 {
        number()
        //~^ ERROR mismatched types
    }
}

fn main() {
    unit()
    //~^ ERROR mismatched types
}
