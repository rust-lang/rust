//[full] run-pass
// Verifies that having generic parameters after constants is not permitted without the
// `const_generics` feature.
// revisions: none min full

#![cfg_attr(full, feature(const_generics))]
#![cfg_attr(full, allow(incomplete_features))]
#![cfg_attr(min, feature(min_const_generics))]

struct A<const N: usize, T=u32>(T);
//[none]~^ ERROR type parameters must be declared prior
//[none]~| ERROR const generics are unstable
//[min]~^^^ ERROR type parameters must be declared prior

fn main() {
  let _: A<3> = A(0);
}
