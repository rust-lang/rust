//@ edition:2021
//@ check-pass
//@ revisions: current next
//@ ignore-compare-mode-next-solver (explicit revisions)
//@[next] compile-flags: -Znext-solver

trait Trait {}
impl Trait for () {}

async fn first_trait() -> impl Trait {
    other_trait().await
}

async fn other_trait() -> impl Trait {
    ()
}

async fn first_iter() -> impl Iterator<Item = ()> {
    other_iter().await
}

async fn other_iter() -> impl Iterator<Item = ()> {
    std::iter::empty()
}

async fn first_future() -> impl std::future::Future<Output = ()> {
    other_future()
}

async fn other_future() {}

fn main() {}
