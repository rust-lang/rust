// edition:2018
// incremental

pub struct SadGirl;

impl SadGirl {
    pub async fn call(&self) -> Result<(), ()> {
        Ok(())
    }
}

async fn async_main() -> Result<(), ()> {
    // should be `.call().await?`
    SadGirl {}.call()?; //~ ERROR: the `?` operator can only be applied to values
    Ok(())
}

fn main() {
    let _ = async_main();
}
