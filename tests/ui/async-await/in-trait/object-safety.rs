// edition:2021


trait Foo {
    async fn foo(&self);
}

fn main() {
    let x: &dyn Foo = todo!();
    //~^ ERROR the trait `Foo` cannot be made into an object
}
