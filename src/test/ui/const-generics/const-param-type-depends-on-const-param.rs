#![feature(const_generics)]
//~^ WARN the feature `const_generics` is incomplete

// Currently, const parameters cannot depend on other generic parameters,
// as our current implementation can't really support this.
//
// We may want to lift this restriction in the future.

pub struct Dependent<const N: usize, const X: [u8; N]>([(); N]);
//~^ ERROR: the type of const parameters must not depend on other generic parameters

pub struct SelfDependent<const N: [u8; N]>;
//~^ ERROR: the type of const parameters must not depend on other generic parameters

fn main() {}
