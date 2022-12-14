// error-pattern: this file contains an unclosed delimiter
// error-pattern: expected value, found struct `R`
struct R { }
struct S {
    x: [u8; R
