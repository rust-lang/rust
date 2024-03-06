//@ edition: 2021

#![feature(return_type_notation)]
//~^ WARN the feature `return_type_notation` is incomplete

trait Trait {
    async fn method() {}
}

fn bar<T: Trait<methid(): Send>>() {}
//~^ ERROR associated function `methid` not found for `Trait`

fn main() {}
