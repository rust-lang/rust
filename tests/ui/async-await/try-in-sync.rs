//@ edition: 2021
#![allow(todo_macro_uses)]

async fn foo() -> Result<(), ()> { todo!() }

fn main() -> Result<(), ()> {
    foo()?;
    //~^ ERROR the `?` operator can only be applied to values that implement `Try`
    Ok(())
}
