// ignore-tidy-trailing-newlines

// Verify that unused parens lint does not try to create a span
// which points in the middle of a multibyte character.

//~v ERROR this file contains an unclosed delimiter
fn f(){(print!(á