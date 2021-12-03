// error-pattern: this file contains an unclosed delimiter
// error-pattern: cannot find type `ţ` in this scope
// error-pattern: parenthesized type parameters may only be used with a `Fn` trait
// error-pattern: type arguments are not allowed for this type
// error-pattern: mismatched types
// ignore-tidy-trailing-newlines
// `ţ` must be the last character in this file, it cannot be followed by a newline
fn main() {
    0: u8(ţ