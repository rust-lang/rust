//@ check-pass
//@ edition:2018

// See issue 60414

trait Trait {
    type Assoc;
}

async fn foo<T: Trait<Assoc=()>>() -> T::Assoc {
    ()
}

fn main() {}
