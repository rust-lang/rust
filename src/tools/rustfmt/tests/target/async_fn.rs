// rustfmt-edition: 2018

async fn bar() -> Result<(), ()> {
    Ok(())
}

pub async fn baz() -> Result<(), ()> {
    Ok(())
}

async unsafe fn foo() {
    async move { Ok(()) }
}

async unsafe fn rust() {
    async move {
        // comment
        Ok(())
    }
}

async fn await_try() {
    something.await?;
}
