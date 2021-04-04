// revisions: full min

#![cfg_attr(full, feature(const_generics))]
#![cfg_attr(full, allow(incomplete_features))]

// Currently, const parameters cannot depend on other generic parameters,
// as our current implementation can't really support this.
//
// We may want to lift this restriction in the future.

pub struct Dependent<T, const X: T>([(); X]);
//~^ ERROR: the type of const parameters must not depend on other generic parameters
//~| ERROR: parameter `T` is never used

fn main() {}
