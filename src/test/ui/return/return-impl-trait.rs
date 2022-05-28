// run-rustfix

trait Trait {}
impl Trait for () {}

// this works
fn foo() -> impl Trait {
    ()
}

fn bar<T: Trait + std::marker::Sync>() -> T
where
    T: Send,
{
    () //~ ERROR mismatched types
}

fn other_bounds<T>() -> T
where
    T: Trait,
    Vec<usize>: Clone,
{
    () //~ ERROR mismatched types
}

fn main() {
    foo();
    bar::<()>();
    other_bounds::<()>();
}
