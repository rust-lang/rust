//@ edition:2018

fn main() {

}

async fn foo() {
    // Adding an .await here avoids the ICE
    test()?;
    //~^ ERROR the `?` operator can only be applied to values that implement `Try`
    //~| ERROR the `?` operator can only be used in an async function that returns
}

// Removing the const generic parameter here avoids the ICE
async fn test<const N: usize>() {
}
