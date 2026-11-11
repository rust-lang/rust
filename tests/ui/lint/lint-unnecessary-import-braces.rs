#![deny(unused_import_braces)]

use crate::test::{A}; //~ ERROR braces around A is unnecessary

mod test {
    use crate::test::{self}; // OK
    use crate::test::{self as rename}; // OK
    pub struct A;
}

fn main() {}
