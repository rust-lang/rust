// edition:2018

async fn foo() -> Result<(), ()> {
    Ok(())
}

async fn bar() -> Result<(), ()> {
    foo()?; //~ ERROR the `?` operator can only be applied to values that implement `std::ops::Try`
    Ok(())
}

fn main() {}
