//[full] run-pass
// Verifies that having generic parameters after constants is not permitted without the
// `const_generics_defaults` feature.
// revisions: min full
#![cfg_attr(full, feature(const_generics_defaults))]

struct A<const N: usize, T=u32>(T);
//[min]~^ ERROR type parameters must be declared prior

fn main() {
    let _: A<3> = A(0);
}
