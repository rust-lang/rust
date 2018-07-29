// rustfmt-edition: Edition2018

async fn bar() -> Result<(), ()> {
    Ok(())
}

pub async fn baz() -> Result<(), ()> {
    Ok(())
}

unsafe async fn foo() {
    async move { Ok(()) }
}
