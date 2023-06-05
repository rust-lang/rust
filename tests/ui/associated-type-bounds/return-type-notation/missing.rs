// edition: 2021

#![feature(return_type_notation, async_fn_in_trait)]
//~^ WARN the feature `return_type_notation` is incomplete

trait Trait {
    async fn method() {}
}

fn bar<T: Trait<methid(): Send>>() {}
//~^ ERROR cannot find associated function `methid` for `Trait`

fn main() {}
