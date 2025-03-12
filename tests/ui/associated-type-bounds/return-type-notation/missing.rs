//@ edition: 2021

trait Trait {
    async fn method() {}
}

fn bar<T: Trait<methid(..): Send>>() {}
//~^ ERROR associated function `methid` not found for `Trait`

fn main() {}
