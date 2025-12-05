//@ edition:2021

async fn foo() -> Result<(), String> {
    Ok(())
}

fn convert_result<T, E>(r: Result<T, E>) -> Option<T> {
    None
}

fn main() -> Option<()> {
    //~^ ERROR `main` has invalid return type `Option<()>`
    convert_result(foo())
    //~^ ERROR mismatched types
}
