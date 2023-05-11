// ignore-tidy-trailing-newlines
//
// error-pattern: this file contains an unclosed delimiter
// error-pattern: this file contains an unclosed delimiter
// error-pattern: this file contains an unclosed delimiter
//
// Verify that unused parens lint does not try to create a span
// which points in the middle of a multibyte character.

fn f(){(print!(á