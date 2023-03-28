// compile-flags: -Zdrop-tracking
// edition: 2021

fn main() {}

async fn foo() {
    None { value: (), ..Default::default() }.await;
    //~^ ERROR variant `Option<_>::None` has no field named `value`
}
