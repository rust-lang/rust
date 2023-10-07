fn foo() -> Result<String, String> { //~ NOTE expected `String` because of this
    let test = String::from("one,two");
    let x = test
        .split_whitespace()
        .next()
        .ok_or_else(|| { //~ NOTE this has type `Result<_, &str>`
            "Couldn't split the test string"
        });
    let one = x
        .map(|s| ())
        .map_err(|_| ()) //~ NOTE this can't be annotated with `?` because it has type `Result<_, ()>`
        .map(|()| "")?; //~ ERROR `?` couldn't convert the error to `String`
    //~^ NOTE in this expansion of desugaring of operator `?`
    //~| NOTE in this expansion of desugaring of operator `?`
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
    Ok(one)
}

fn baz() -> Result<String, String> { //~ NOTE expected `String` because of this
    let test = String::from("one,two");
    let one = test
        .split_whitespace()
        .next()
        .ok_or_else(|| { //~ NOTE this can't be annotated with `?` because it has type `Result<_, ()>`
            "Couldn't split the test string";
        })?;
    //~^ ERROR `?` couldn't convert the error to `String`
    //~| NOTE in this expansion of desugaring of operator `?`
    //~| NOTE in this expansion of desugaring of operator `?`
    //~| NOTE in this expansion of desugaring of operator `?`
    //~| NOTE in this expansion of desugaring of operator `?`
    //~| NOTE the trait `From<()>` is not implemented for `String`
    //~| NOTE the question mark operation (`?`) implicitly performs a conversion on the error value using the `From` trait
    //~| NOTE required for `Result<String, String>` to implement `FromResidual<Result<Infallible, ()>>`
    Ok(one.to_string())
}

fn main() {}
