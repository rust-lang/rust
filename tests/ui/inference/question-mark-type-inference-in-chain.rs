// #129269
use std::fmt::Display;

#[derive(Debug)]
struct AnotherError;
type Result<T, E = AnotherError> = core::result::Result<T, E>;

#[derive(Debug)]
pub struct Error;

impl From<AnotherError> for Error {
    fn from(_: AnotherError) -> Self { Error }
}

impl std::error::Error for Error {}

impl Display for Error {
    fn fmt(&self, _f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        todo!()
    }
}

#[derive(Ord, PartialEq, PartialOrd, Eq)]
pub struct Version {}

fn parse(_s: &str) -> std::result::Result<Version, Error> {
    todo!()
}

pub fn error1(lines: &[&str]) -> Result<Vec<Version>> {
    let mut tags = lines.iter().map(|e| parse(e)).collect()?;
    //~^ ERROR: type annotations needed
    //~| HELP: consider giving `tags` an explicit type

    tags.sort(); //~ NOTE: type must be known at this point

    Ok(tags)
}

pub fn error2(lines: &[&str]) -> Result<Vec<Version>> {
    let mut tags: Vec<Version> = lines.iter().map(|e| parse(e)).collect()?;
    //~^ ERROR: type annotations needed
    //~| NOTE: cannot infer type of the type parameter `B`
    //~| NOTE: the type must implement `FromIterator<std::result::Result<Version, Error>>`
    //~| NOTE: required by a bound in `collect`
    //~| HELP: consider specifying the generic argument
    tags.sort();

    Ok(tags)
}

pub fn error3(lines: &[&str]) -> Result<Vec<Version>> {
    let mut tags = lines.iter().map(|e| parse(e)).collect::<Vec<_>>()?;
    //~^ ERROR: the `?` operator can only be applied to values that implement `Try`
    //~| NOTE: the `?` operator cannot be applied to type `Vec<std::result::Result<Version, Error>>`
    //~| HELP: the nightly-only, unstable trait `Try` is not implemented
    //~| NOTE: in this expansion of desugaring of operator `?`
    //~| NOTE: in this expansion of desugaring of operator `?`
    tags.sort();

    Ok(tags)
}

pub fn error4(lines: &[&str]) -> Result<Vec<Version>> {
    let mut tags = lines
        //~^ NOTE: this expression has type `&[&str]`
        .iter()
        //~^ NOTE: `Iterator::Item` is `&&str` here
        .map(|e| parse(e))
        //~^ NOTE: the method call chain might not have had the expected associated types
        //~| NOTE: `Iterator::Item` changed to `Result<Version, Error>` here
        .collect::<Result<Vec<Version>>>()?;
        //~^ ERROR: a value of type `std::result::Result<Vec<Version>, AnotherError>` cannot be built from an iterator over elements of type `std::result::Result<Version, Error>`
        //~| NOTE: value of type `std::result::Result<Vec<Version>, AnotherError>` cannot be built from `std::iter::Iterator<Item=std::result::Result<Version, Error>>`
        //~| NOTE: required by a bound introduced by this call
        //~| HELP: the trait
        //~| HELP: for that trait implementation, expected `AnotherError`, found `Error`
        //~| NOTE: required by a bound in `collect`
    tags.sort();

    Ok(tags)
}

fn main() {}
