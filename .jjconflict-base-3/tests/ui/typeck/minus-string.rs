// Regression test for issue #813.
// This ensures that the unary negation operator `-` cannot be applied to an owned `String`.
// Previously, due to a type-checking bug, this was mistakenly accepted by the compiler.

fn main() {
    -"foo".to_string(); //~ ERROR cannot apply unary operator `-` to type `String`
}
