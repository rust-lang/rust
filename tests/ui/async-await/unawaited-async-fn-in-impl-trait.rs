//@ edition:2021
//@ revisions: current next
//@ ignore-compare-mode-next-solver (explicit revisions)
//@[next] compile-flags: -Znext-solver

trait Trait {}
impl Trait for () {}

async fn first_trait() -> impl Trait {
    //[current]~^ ERROR the trait bound `impl Future<Output = impl Trait>: Trait` is not satisfied
    //[current]~^^ ERROR the trait bound `impl Future<Output = impl Trait>: Trait` is not satisfied
    //[next]~^^^ ERROR the trait bound `impl Future<Output = impl Trait>: Trait` is not satisfied
    other_trait()
}

async fn other_trait() -> impl Trait {
    ()
}

async fn first_iter() -> impl Iterator<Item = ()> {
    //[current]~^ ERROR `impl Future<Output = impl Iterator<Item = ()>>` is not an iterator
    //[current]~^^ ERROR `impl Future<Output = impl Iterator<Item = ()>>` is not an iterator
    //[next]~^^^ ERROR `impl Future<Output = impl Iterator<Item = ()>>` is not an iterator
    other_iter()
}

async fn other_iter() -> impl Iterator<Item = ()> {
    std::iter::empty()
}

fn main() {}
