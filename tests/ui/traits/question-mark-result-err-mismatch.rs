#![feature(try_trait_v2)]
fn foo() -> Result<String, String> { //~ NOTE expected `String` because of this
    let test = String::from("one,two");
    let x = test
        .split_whitespace()
        .next()
        .ok_or_else(|| {
            "Couldn't split the test string"
        });
    let one = x
        .map(|s| ())
        .map_err(|e| { //~ NOTE this can't be annotated with `?` because it has type `Result<_, ()>`
            e; //~ HELP remove this semicolon
        })
        .map(|()| "")?; //~ ERROR `?` couldn't convert the error to `String`
    //~^ NOTE in this expansion of desugaring of operator `?`
    //~| NOTE in this expansion of desugaring of operator `?`
    //~| NOTE in this expansion of desugaring of operator `?`
    //~| NOTE the trait `From<()>` is not implemented for `String`
    //~| NOTE the question mark operation (`?`) implicitly performs a conversion on the error value using the `From` trait
    //~| NOTE required for `Result<String, String>` to implement `FromResidual<Result<Infallible, ()>>`
    Ok(one.to_string())
}

fn bar() -> Result<(), String> { //~ NOTE expected `String` because of this
    let x = foo(); //~ NOTE this has type `Result<_, String>`
    let one = x
        .map(|s| ())
        .map_err(|_| ())?; //~ ERROR `?` couldn't convert the error to `String`
    //~^ NOTE in this expansion of desugaring of operator `?`
    //~| NOTE in this expansion of desugaring of operator `?`
    //~| NOTE in this expansion of desugaring of operator `?`
    //~| NOTE in this expansion of desugaring of operator `?`
    //~| NOTE this can't be annotated with `?` because it has type `Result<_, ()>`
    //~| NOTE the trait `From<()>` is not implemented for `String`
    //~| NOTE the question mark operation (`?`) implicitly performs a conversion on the error value using the `From` trait
    //~| NOTE required for `Result<(), String>` to implement `FromResidual<Result<Infallible, ()>>`
    //~| HELP the following other types implement trait `From<T>`:
    Ok(one)
}

fn baz() -> Result<String, String> { //~ NOTE expected `String` because of this
    let test = String::from("one,two");
    let one = test
        .split_whitespace()
        .next()
        .ok_or_else(|| { //~ NOTE this can't be annotated with `?` because it has type `Result<_, ()>`
            "Couldn't split the test string"; //~ HELP remove this semicolon
        })?;
    //~^ ERROR `?` couldn't convert the error to `String`
    //~| NOTE in this expansion of desugaring of operator `?`
    //~| NOTE in this expansion of desugaring of operator `?`
    //~| NOTE in this expansion of desugaring of operator `?`
    //~| NOTE the trait `From<()>` is not implemented for `String`
    //~| NOTE the question mark operation (`?`) implicitly performs a conversion on the error value using the `From` trait
    //~| NOTE required for `Result<String, String>` to implement `FromResidual<Result<Infallible, ()>>`
    Ok(one.to_string())
}

fn main() {}
