// Make sure the #[async_send] attribute requires the async_fn_in_trait feature

// edition: 2021

trait MyTrait {
    #[async_send] //~ `async_send` is a temporary placeholder
    async fn foo(&self) -> usize;
    //~^ functions in traits cannot be declared `async`
}

#[async_send] //~ `async_send` is a temporary placeholder
async fn bar() {}

fn main() {}
