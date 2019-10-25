#![feature(const_generics)]
//~^ WARN the feature `const_generics` is incomplete and may cause the compiler to crash

// Currently, const parameters cannot depend on type parameters, because there is no way to
// enforce the `structural_match` property on an arbitrary type parameter. This restriction
// may be relaxed in the future. See https://github.com/rust-lang/rfcs/pull/2000 for more
// details.

pub struct Dependent<T, const X: T>([(); X]);
//~^ ERROR const parameters cannot depend on type parameters
//~^^ ERROR parameter `T` is never used

fn main() {}
