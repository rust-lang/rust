// edition: 2021

trait Foo {
    type T;

    async fn foo(&self) -> Self::T;
}

struct Bar;

impl Foo for Bar {
    type T = ();

    async fn foo(&self) {}
    //~^ ERROR type annotations needed: cannot satisfy `<Bar as Foo>::T == ()`
}

impl Foo for Bar {
    //~^ ERROR conflicting implementations of trait `Foo` for type `Bar`
    type T = ();

    async fn foo(&self) {}
    //~^ ERROR type annotations needed: cannot satisfy `<Bar as Foo>::T == ()`
}

fn main() {}
