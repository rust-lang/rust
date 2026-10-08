//@ edition:2018

fn main() {

}

async fn foo() {
    // Adding an .await here avoids the ICE
    test()?;
    //~^ ERROR the trait bound `impl Future<Output = ()>: ops::try_trait_old::Try` is not satisfied [E0277]
    //~| ERROR the `?` operator can only be applied to values that implement `ops::try_trait_old::Try` [E0277]
    //~| ERROR the `?` operator can only be used in an async function that returns `Result` or `Option` (or another type that implements `ops::try_trait_old::FromResidual`) [E0277]
}

// Removing the const generic parameter here avoids the ICE
async fn test<const N: usize>() {
}
