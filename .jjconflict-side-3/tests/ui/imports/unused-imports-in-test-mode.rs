//@ compile-flags: --test

#![deny(unused_imports)]

use std::io::BufRead; //~ ERROR unused import: `std::io::BufRead`

fn a() {}
fn b() {}

mod test {
    use super::a;

    #[test]
    fn foo() {
        a();
        use crate::b; //~ ERROR unused import: `crate::b`
    }
}

mod tests {
    use super::a;

    #[test]
    fn foo() {
        a();
        use crate::b; //~ ERROR unused import: `crate::b`
    }
}

mod test_a {
    use super::a;

    #[test]
    fn foo() {
        a();
        use crate::b; //~ ERROR unused import: `crate::b`
    }
}

mod a_test {
    use super::a;

    #[test]
    fn foo() {
        a();
        use crate::b; //~ ERROR unused import: `crate::b`
    }
}

mod tests_a {
    use super::a;

    #[test]
    fn foo() {
        a();
        use crate::b; //~ ERROR unused import: `crate::b`
    }
}

mod a_tests {
    use super::a;

    #[test]
    fn foo() {
        a();
        use crate::b; //~ ERROR unused import: `crate::b`
    }
}

mod fastest_search {
    use super::a;

    #[test]
    fn foo() {
        a();
        use crate::b; //~ ERROR unused import: `crate::b`
    }
}

#[cfg(test)]
mod test_has_attr {
    use super::a;

    #[test]
    fn foo() {
        a();
        use crate::b; //~ ERROR unused import: `crate::b`
    }
}

mod test_has_no_attr {
    #[cfg(test)]
    use super::a;

    #[test]
    fn foo() {
        a();
        use crate::b; //~ ERROR unused import: `crate::b`
    }
}

fn main() {}
