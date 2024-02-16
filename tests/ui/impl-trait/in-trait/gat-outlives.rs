//@ edition: 2021

use std::future::Future;

trait Trait {
    type Gat<'a>;
    //~^ ERROR missing required bound on `Gat`
    async fn foo(&self) -> Self::Gat<'_>;
}

trait Trait2 {
    type Gat<'a>;
    //~^ ERROR missing required bound on `Gat`
    async fn foo(&self) -> impl Future<Output = Self::Gat<'_>>;
}

fn main() {}
