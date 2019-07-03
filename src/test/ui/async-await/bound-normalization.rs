// check-pass
// edition:2018

#![feature(async_await)]

// See issue 60414

trait Trait {
    type Assoc;
}

async fn foo<T: Trait<Assoc=()>>() -> T::Assoc {
    ()
}

fn main() {}
