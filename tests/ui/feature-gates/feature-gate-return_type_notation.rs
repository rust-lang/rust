//@ edition: 2021
//@ revisions: cfg no

trait Trait {
    #[allow(async_fn_in_trait)]
    async fn m();
}

#[cfg(cfg)]
fn foo<T: Trait<m(..): Send>>() {}
//~^ ERROR return type notation is experimental

fn main() {}
