#![feature(non_lifetime_binders)]

fn f()
where
    for<B> B::Item: Send,
    //~^ ERROR ambiguous associated type
{
}

fn main() {}
