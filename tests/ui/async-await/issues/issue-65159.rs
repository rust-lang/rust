// Regression test for #65159. We used to ICE.
//
// edition:2018

async fn copy() -> Result<()>
//~^ ERROR enum takes 2 generic arguments
{
    Ok(())
}

fn main() { }
