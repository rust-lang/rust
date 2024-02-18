//@ [full] check-pass
//@ revisions: full min
#![cfg_attr(full, feature(adt_const_params))]
#![cfg_attr(full, allow(incomplete_features))]

struct Foo<const V: [usize; 0] > {}
//[min]~^ ERROR `[usize; 0]` is forbidden as the type of a const generic parameter

type MyFoo = Foo<{ [] }>;

fn main() {
    let _ = Foo::<{ [] }> {};
}
