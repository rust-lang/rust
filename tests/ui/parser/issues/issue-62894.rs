// Regression test for #62894, shouldn't crash.
// error-pattern: this file contains an unclosed delimiter

fn f() { assert_eq!(f(), (), assert_eq!(assert_eq!

fn main() {}
