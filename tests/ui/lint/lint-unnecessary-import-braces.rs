#![deny(unused_import_braces)]

use test::{A}; //~ ERROR braces around A is unnecessary

mod test {
    use test::{self}; // OK
    use test::{self as rename}; // OK
    pub struct A;
}

fn main() {}
