// Here, there are two types with the same name. One of these has a `derive` annotation, but in the
// expansion these `impl`s are associated to the *other* type. There is a suggestion to remove
// unneeded type parameters, but because we're now point at a type with no type parameters, the
// suggestion would suggest removing code from an empty span, which would ICE in nightly.
//
// issue: rust-lang/rust#108748

struct NotSM;

#[derive(PartialEq, Eq)]
//~^ ERROR struct takes 0 generic arguments
//~| ERROR struct takes 0 generic arguments
//~| ERROR struct takes 0 generic arguments
//~| ERROR struct takes 0 generic arguments
struct NotSM<T>(T);
//~^ ERROR the name `NotSM` is defined multiple times
//~| ERROR no field `0`

fn main() {}
