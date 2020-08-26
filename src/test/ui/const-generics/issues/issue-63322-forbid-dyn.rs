// revisions: full min
#![cfg_attr(full, feature(const_generics))]
#![cfg_attr(full, allow(incomplete_features))]
#![cfg_attr(min, feature(min_const_generics))]

trait A {}
struct B;
impl A for B {}

fn test<const T: &'static dyn A>() {
    //[full]~^ ERROR must be annotated with `#[derive(PartialEq, Eq)]` to be used
    //[min]~^^ ERROR `&'static (dyn A + 'static)` is forbidden as the type of a const generic parameter
    //[min]~| ERROR must be annotated with `#[derive(PartialEq, Eq)]` to be used
    unimplemented!()
}

fn main() {
    test::<{ &B }>();
}
