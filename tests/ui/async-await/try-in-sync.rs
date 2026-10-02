//@ edition: 2021

async fn foo() -> Result<(), ()> { todo!() }

fn main() -> Result<(), ()> {
    foo()?;
    //~^ ERROR the trait bound `impl Future<Output = Result<(), ()>>: ops::try_trait_old::Try` is not satisfied [E0277]
    //~| ERROR the `?` operator can only be applied to values that implement `ops::try_trait_old::Try` [E0277]
    Ok(())
}
