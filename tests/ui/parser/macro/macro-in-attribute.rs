// Test for #146325.
// Ensure that when we encounter a macro invocation in an attribute, we don't suggest nonsense.

#[deprecated(note = concat!("a", "b"))]
struct X;
//~^^ ERROR: expected a literal (`1u8`, `1.0f32`, `"string"`, etc.) here, found
//~| NOTE: macro calls are not allowed here

fn main() {}
