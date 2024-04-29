// Ensures to suggest quoting attribute values only when they are identifiers.

#[doc(alias = "val")]
#[doc(alias = val)] //~ ERROR expected unsuffixed literal, found `val`
#[doc(alias = ["va", "al"])] //~ ERROR expected unsuffixed literal or identifier, found `[`
#[doc(alias = &["va", "al"])] //~ ERROR expected unsuffixed literal or identifier, found `&`
fn main() {}
