// Ensure that we get an error and not an ICE for this problematic case.
struct Foo<T = Option<U>, U = bool>(T, U);
//~^ ERROR type parameters with a default cannot use forward declared identifiers
fn main() {
    let x: Foo;
}
