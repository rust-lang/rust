//@ edition: 2021

trait Foo {
    async fn bar(&self);
}

async fn takes_dyn_trait(x: &dyn Foo) {
    //~^ ERROR the trait `Foo` is not dyn compatible
    x.bar().await;
}

fn main() {}
