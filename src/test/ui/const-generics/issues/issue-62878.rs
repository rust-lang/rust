// revisions: full min
#![cfg_attr(full, feature(const_generics))]
#![cfg_attr(full, allow(incomplete_features))]
#![cfg_attr(min, feature(min_const_generics))]

fn foo<const N: usize, const A: [u8; N]>() {}
//~^ ERROR the type of const parameters must not
//[min]~| ERROR `[u8; _]` is forbidden as the type of a const generic parameter

fn main() {
    foo::<_, {[1]}>();
    //[full]~^ ERROR type provided when a constant was expected
    //[full]~| ERROR mismatched types
}
