// revisions: full min

#![cfg_attr(full, feature(const_generics))]
#![cfg_attr(full, allow(incomplete_features))]

// Currently, const parameters cannot depend on other generic parameters,
// as our current implementation can't really support this.
//
// We may want to lift this restriction in the future.

pub struct Dependent<const N: usize, const X: [u8; N]>([(); N]);
//~^ ERROR: the type of const parameters must not depend on other generic parameters
//[min]~^^ ERROR `[u8; _]` is forbidden

pub struct SelfDependent<const N: [u8; N]>;
//~^ ERROR: the type of const parameters must not depend on other generic parameters
//[min]~^^ ERROR `[u8; _]` is forbidden

fn main() {}
