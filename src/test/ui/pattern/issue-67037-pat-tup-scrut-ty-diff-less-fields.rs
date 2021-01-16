// Regression test for #67037.
//
// In type checking patterns, E0023 occurs when the tuple pattern and the expected
// tuple pattern have different number of fields. For example, as below, `P()`,
// the tuple struct pattern, has 0 fields, but requires 1 field.
//
// In emitting E0023, we try to see if this is a case of e.g., `Some(a, b, c)` but where
// the scrutinee was of type `Some((a, b, c))`, and suggest that parenthesis be added.
//
// However, we did not account for the expected type being different than the tuple pattern type.
// This caused an issue when the tuple pattern type (`P<T>`) was generic.
// Specifically, we tried deriving the 0th field's type using the `substs` of the expected type.
// When attempting to substitute `T`, there was no such substitution, so "out of range" occurred.

struct U {} // 0 type parameters offered
struct P<T>(T); // 1 type parameter wanted

fn main() {
    let P() = U {}; //~ ERROR mismatched types
    //~^ ERROR this pattern has 0 fields, but the corresponding tuple struct has 1 field
}
