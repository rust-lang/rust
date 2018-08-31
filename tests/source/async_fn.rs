// rustfmt-edition: 2018

async fn bar() -> Result<(), ()> {
    Ok(())
}

pub async fn baz() -> Result<(), ()> {
    Ok(())
}

unsafe async fn foo() {
    async move {
        Ok(())
    }
}

unsafe async fn rust() {
    async move { // comment
        Ok(())
    }
}
