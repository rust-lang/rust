//@ edition: 2021

trait Foo {
    async fn bar(&self);
}

async fn takes_dyn_trait(x: &dyn Foo) {
    //~^ ERROR the trait `Foo` cannot be made into an object
    x.bar().await;
    //~^ ERROR the trait `Foo` cannot be made into an object
    //~| ERROR the trait `Foo` cannot be made into an object
}

fn main() {}
