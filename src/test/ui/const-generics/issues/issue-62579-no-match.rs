// [full] run-pass
// revisions: full min
#![cfg_attr(full, feature(const_generics))]
#![cfg_attr(full, allow(incomplete_features))]

#[derive(PartialEq, Eq)]
struct NoMatch;

fn foo<const T: NoMatch>() -> bool {
    //[min]~^ ERROR `NoMatch` is forbidden as the type of a const generic parameter
    true
}

fn main() {
    foo::<{NoMatch}>();
}
