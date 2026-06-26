//@ edition: 2021
//@ run-rustfix

#![allow(unused)]

struct Foo;

impl Foo {
    fn get(&self) -> u8 {
        42
    }
}

fn test_result_in_result() -> Result<(), ()> {
    let res: Result<_, ()> = Ok(Foo);
    res.get();
    //~^ ERROR no method named `get` found for enum `Result<T, E>` in the current scope
    //~| HELP use the `?` operator
    Ok(())
}

async fn async_test_result_in_result() -> Result<(), ()> {
    let res: Result<_, ()> = Ok(Foo);
    res.get();
    //~^ ERROR no method named `get` found for enum `Result<T, E>` in the current scope
    //~| HELP use the `?` operator
    Ok(())
}

fn test_result_in_unit_return() {
    let res: Result<_, ()> = Ok(Foo);
    res.get();
    //~^ ERROR no method named `get` found for enum `Result<T, E>` in the current scope
    //~| HELP consider using `Result::expect` to unwrap the `Foo` value, panicking if the value is a `Result::Err`
}

async fn async_test_result_in_unit_return() {
    let res: Result<_, ()> = Ok(Foo);
    res.get();
    //~^ ERROR no method named `get` found for enum `Result<T, E>` in the current scope
    //~| HELP consider using `Result::expect` to unwrap the `Foo` value, panicking if the value is a `Result::Err`
}

fn test_option_in_option() -> Option<()> {
    let res: Option<_> = Some(Foo);
    res.get();
    //~^ ERROR no method named `get` found for enum `Option<T>` in the current scope
    //~| HELP use the `?` operator
    Some(())
}

fn test_option_in_unit_return() {
    let res: Option<_> = Some(Foo);
    res.get();
    //~^ ERROR no method named `get` found for enum `Option<T>` in the current scope
    //~| HELP consider using `Option::expect` to unwrap the `Foo` value, panicking if the value is an `Option::None`
}

fn test_option_private_method() {
    let res: Option<_> = Some(vec![1, 2, 3]);
    res.len();
    //~^ ERROR method `len` is private
    //~| HELP consider using `Option::expect` to unwrap the `Vec<{integer}>` value, panicking if the value is an `Option::None`
}

fn main() {}
