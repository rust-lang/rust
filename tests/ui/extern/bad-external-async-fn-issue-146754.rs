//@ edition:2024
#![crate_type = "lib"]

unsafe extern "C" {
    async fn function() -> [(); || {}];
    //~^ ERROR functions in `extern` blocks cannot have `async` qualifier
    //~^^ ERROR mismatched types
}
