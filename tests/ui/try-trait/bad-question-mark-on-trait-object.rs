struct E;
//~^ HELP the trait `std::error::Error` is not implemented for `E`
struct X;

fn foo() -> Result<(), Box<dyn std::error::Error>> {
    Ok(bar()?)
    //~^ ERROR the trait bound `E: std::error::Error` is not satisfied [E0277]
    //~| NOTE unsatisfied trait bound
    //~| NOTE required for `Result<(), Box<dyn std::error::Error>>` to implement `ops::try_trait_old::FromResidual<Result<!, E>>`
    //~| NOTE required for `Box<dyn std::error::Error>` to implement `From<E>`
    //~| NOTE in this expansion of desugaring of operator `?`
    //~| NOTE in this expansion of desugaring of operator `?`
    //~| NOTE in this expansion of desugaring of operator `?`
    //~| NOTE in this expansion of desugaring of operator `?`
    //~| HELP the trait `ops::try_trait_old::FromResidual<Result<!, E>>` is conditionally implemented for `Result<T, F>`
}
fn bat() -> Result<(), X> { //~ NOTE expected `X` because of this
    Ok(bar()?)
    //~^ ERROR `?` couldn't convert the error to `X`
    //~| NOTE unsatisfied trait bound
    //~| NOTE in this expansion of desugaring of operator `?`
    //~| NOTE in this expansion of desugaring of operator `?`
    //~| NOTE in this expansion of desugaring of operator `?`
    //~| NOTE in this expansion of desugaring of operator `?`
    //~| NOTE in this expansion of desugaring of operator `?`
    //~| NOTE the question mark operation (`?`) implicitly performs a conversion on the error value using the `From` trait
    //~| NOTE required for `Result<(), X>` to implement `ops::try_trait_old::FromResidual<Result<!, E>>`
    //~| HELP the trait `FromResidual<Result<_, E>>` is not implemented for `Result<(), X>`
    //~| HELP for that trait implementation, expected `X`, found `E`

}
fn bar() -> Result<(), E> {
    Err(E)
}
fn main() {}
