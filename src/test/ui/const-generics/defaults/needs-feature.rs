// Verifies that having generic parameters after constants is not permitted without the
// `const_generics` feature.

struct A<const N: usize, T=u32>(T);
//~^ ERROR type parameters must be declared prior
//~| ERROR const generics are unstable

fn main() {}
