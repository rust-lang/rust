#![feature(impl_trait_in_assoc_type)]

trait Bar {
    type E;
}
impl<S> Bar for S {
    type E = impl ; //~ ERROR at least one trait must be specified
    fn foo() -> Self::E {
        //~^ ERROR `foo` is not a member
        |_| true //~ ERROR type annotations needed
    }
}
fn main() {}
