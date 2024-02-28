trait Future {
    type Item;
    type Error;
}

use std::error::Error;

fn foo() -> impl Future<Item=(), Error=Box<dyn Error>> {
    //~^ ERROR the trait `Future` is not implemented for `Result<(), _>`
    Ok(())
}

fn main() {}
