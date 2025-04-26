//@ edition:2018

async fn test() -> Result<(), Box<dyn std::error::Error>> {
    macro!();
    //~^ ERROR expected identifier, found `!`
    Ok(())
}

fn main() {}
