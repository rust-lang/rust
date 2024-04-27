trait Future {
    type Item;
    type Error;
}

use std::error::Error;

fn foo() -> impl Future<Item=(), Error=Box<dyn Error>> {
    //~^ ERROR not satisfied
    Ok(())
}

fn main() {}
