// Test for #146325.
// Ensure that when we encounter an expr invocation in an attribute, we don't suggest nonsense.

#[deprecated(note = a!=b)]
struct X;
//~^^ ERROR: expected a literal (`1u8`, `1.0f32`, `"string"`, etc.) here, found expression
//~| NOTE: expressions are not allowed here

fn main() {}
