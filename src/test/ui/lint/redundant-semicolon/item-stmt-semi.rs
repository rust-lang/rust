// check-pass
// This test should stop compiling
// we decide to enable this lint for item statements.

#![deny(redundant_semicolons)]

fn main() {
    fn inner() {};
    struct Bar {};
}
