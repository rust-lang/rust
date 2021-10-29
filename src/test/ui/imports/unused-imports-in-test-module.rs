#![deny(unused_imports)]

use std::io::BufRead; //~ ERROR unused import: `std::io::BufRead`

fn a() {}
fn b() {}

mod test {
    use super::a;  //~ ERROR unused import: `super::a`

    fn foo() {
        use crate::b;  //~ ERROR unused import: `crate::b`
    }
}

mod test_a {
    use super::a;  //~ ERROR unused import: `super::a`

    fn foo() {
        use crate::b;  //~ ERROR unused import: `crate::b`
    }
}

fn main() {}
