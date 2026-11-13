// The expression parser may sometimes emit early lints during late attribute parsing.
// Those will never be emitted because the early lint buffer is already emptied at that time.
// This is fine, as long as this only happens for code that is invalid.

#[feature({fn a(...) })]
//~^ ERROR expected one of `->`, `where`, or `{`, found `}`
//~| ERROR expected a literal (`1u8`, `1.0f32`, `"string"`, etc.) here, found expression
fn main() {}
