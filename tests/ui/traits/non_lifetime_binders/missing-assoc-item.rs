#![feature(non_lifetime_binders)]
//~^ WARN the feature `non_lifetime_binders` is incomplete

fn f()
where
    for<B> B::Item: Send,
    //~^ ERROR ambiguous associated type
{
}

fn main() {}
