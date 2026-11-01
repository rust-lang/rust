//@ edition:2018
use core::future::Future;

async fn base_thing() -> Result<(), ()> {
    Ok(())
}

fn thing_one() -> impl Future<Output = Result<(), ()>> {
    base_thing()
}

fn thing_two() -> impl Future<Output = Result<(), ()>> {
    base_thing()
}

async fn thing() -> Result<(), ()> {
    if true {
        thing_one()
    } else {
        thing_two() //~ ERROR `if` and `else` have incompatible types
    }.await
}

fn main() {}
