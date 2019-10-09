// Regression test for #65159. We used to ICE.
//
// edition:2018

async fn copy() -> Result<()> //~ ERROR wrong number of type arguments
{
    Ok(())
}

fn main() { }
