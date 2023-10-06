fn foo() -> Result<String, String> { //~ NOTE expected `String` because of this
    let test = String::from("one,two");
    let x = test
        .split_whitespace()
        .next()
        .ok_or_else(|| { //~ NOTE this can be annotated with `?` because it has type `Result<&str, &str>`
            "Couldn't split the test string"
        });
    let one = x
        .map(|s| ()) //~ NOTE this can be annotated with `?` because it has type `Result<(), &str>`
        .map_err(|_| ()) //~ NOTE this can't be annotated with `?` because it has type `Result<(), ()>`
        .map(|()| "")?; //~ ERROR `?` couldn't convert the error to `String`
    //~^ NOTE in this expansion of desugaring of operator `?`
    //~| NOTE in this expansion of desugaring of operator `?`
    //~| NOTE in this expansion of desugaring of operator `?`
    //~| NOTE in this expansion of desugaring of operator `?`
    //~| NOTE this can't be annotated with `?` because it has type `Result<&str, ()>`
    //~| NOTE the trait `From<()>` is not implemented for `String`
    //~| NOTE the question mark operation (`?`) implicitly performs a conversion on the error value using the `From` trait
    //~| NOTE required for `Result<String, String>` to implement `FromResidual<Result<Infallible, ()>>`
    Ok(one.to_string())
}

fn bar() -> Result<(), String> { //~ NOTE expected `String` because of this
    let x = foo(); //~ NOTE this can be annotated with `?` because it has type `Result<String, String>`
    let one = x
        .map(|s| ()) //~ NOTE this can be annotated with `?` because it has type `Result<(), String>`
        .map_err(|_| ())?; //~ ERROR `?` couldn't convert the error to `String`
    //~^ NOTE in this expansion of desugaring of operator `?`
    //~| NOTE in this expansion of desugaring of operator `?`
    //~| NOTE in this expansion of desugaring of operator `?`
    //~| NOTE in this expansion of desugaring of operator `?`
    //~| NOTE this can't be annotated with `?` because it has type `Result<(), ()>`
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
        .ok_or_else(|| { //~ NOTE this can't be annotated with `?` because it has type `Result<&str, ()>`
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
