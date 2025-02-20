struct E;
//~^ NOTE `E` needs to implement `std::error::Error`
//~| NOTE alternatively, `E` needs to implement `Into<X>`
struct X; //~ NOTE `X` needs to implement `From<E>`

fn foo() -> Result<(), Box<dyn std::error::Error>> { //~ NOTE required `E: std::error::Error` because of this
    Ok(bar()?)
    //~^ ERROR `?` couldn't convert the error: `E: std::error::Error` is not satisfied
    //~| NOTE the trait `std::error::Error` is not implemented for `E`
    //~| NOTE the question mark operation (`?`) implicitly performs a conversion on the error value using the `From` trait
    //~| NOTE required for `Box<dyn std::error::Error>` to implement `From<E>`
    //~| NOTE this has type `Result<_, E>`
    //~| NOTE in this expansion
    //~| NOTE in this expansion
    //~| NOTE in this expansion
}
fn bat() -> Result<(), X> { //~ NOTE expected `X` because of this
    Ok(bar()?)
    //~^ ERROR `?` couldn't convert the error to `X`
    //~| NOTE the trait `From<E>` is not implemented for `X`
    //~| NOTE this can't be annotated with `?` because it has type `Result<_, E>`
    //~| NOTE the question mark operation (`?`) implicitly performs a conversion on the error value using the `From` trait
    //~| NOTE in this expansion
    //~| NOTE in this expansion
}
fn bar() -> Result<(), E> {
    Err(E)
}
fn main() {}
