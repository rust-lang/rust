//@ edition: 2021

fn main() {}

async fn foo() {
    None { value: (), ..Default::default() }.await;
    //~^ ERROR `Option<_>` is not a future
    //~| ERROR variant `Option<_>::None` has no field named `value`
}
