// edition:2018

async fn main() -> Result<i32, ()> {
//~^ ERROR `main` function is not allowed to be `async`
//~^^ ERROR `main` has invalid return type `impl std::future::Future`
    Ok(1)
}
