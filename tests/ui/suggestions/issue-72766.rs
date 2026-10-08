//@ edition:2018
//@ incremental

pub struct SadGirl;

impl SadGirl {
    pub async fn call(&self) -> Result<(), ()> {
        Ok(())
    }
}

async fn async_main() -> Result<(), ()> {
    // should be `.call().await?`
    SadGirl {}.call()?; //~ ERROR: the `?` operator can only be applied to values
    //~| ERROR the trait bound `impl Future<Output = Result<(), ()>>: ops::try_trait_old::Try` is not satisfied [E0277]
    Ok(())
}

fn main() {
    let _ = async_main();
}
