#![allow(unused)]

fn foo<T: Default>() -> Result<T, ()> {
    Err(())
}

fn test() -> Result<(), ()> {
    _ = foo()?;
    //~^ error: the trait bound `!: Default` is not satisfied
    Ok(())
}

fn main() {}
