// edition:2018

async fn foo() -> Result<(), ()> {
    Ok(())
}

async fn bar() -> Result<(), ()> {
    foo()?;
    Ok(())
}

fn main() {}
