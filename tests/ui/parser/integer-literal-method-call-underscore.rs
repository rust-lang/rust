//! Checks that methods with names starting with an underscore (`_`) can be
//! successfully called directly on integer literals, confirming the correct
//! parsing of such expressions where the underscore is part of the method identifier.

//@ run-pass

trait Tr: Sized {
    fn _method_on_numbers(self) {}
}

impl Tr for i32 {}

fn main() {
    42._method_on_numbers();
}
