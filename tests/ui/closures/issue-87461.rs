// Regression test for #87461.

//@ edition:2021

async fn func() -> Result<u16, u64> {
    let _ = async {
        Err(42u64)
    }.await?;

    Ok(())
    //~^ ERROR: mismatched types [E0308]
}

async fn func2() -> Result<u16, u64> {
    Err(42u64)?;

    Ok(())
    //~^ ERROR: mismatched types [E0308]
}

fn main() {
    || -> Result<u16, u64> {
        if true {
            return Err(42u64);
        }
        Ok(())
        //~^ ERROR: mismatched types [E0308]
    };
}
