// revisions: full min
#![cfg_attr(full, feature(const_generics))]
#![cfg_attr(full, allow(incomplete_features))]

trait A {}
struct B;
impl A for B {}

fn test<const T: &'static dyn A>() {
    //~^ ERROR must be annotated with `#[derive(PartialEq, Eq)]` to be used
    //[min]~^^ ERROR `&'static (dyn A + 'static)` is forbidden
    unimplemented!()
}

fn main() {
    test::<{ &B }>();
}
