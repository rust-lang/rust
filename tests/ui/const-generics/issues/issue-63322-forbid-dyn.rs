//@ revisions: full min
#![cfg_attr(full, feature(adt_const_params, unsized_const_params))]
#![cfg_attr(full, allow(incomplete_features))]

trait A {}
struct B;
impl A for B {}

fn test<const T: &'static dyn A>() {
    //[full]~^ ERROR `&'static (dyn A + 'static)` can't be used as a const parameter type
    //[min]~^^ ERROR `&'static (dyn A + 'static)` is forbidden
    unimplemented!()
}

fn main() {
    test::<{ &B }>();
}
