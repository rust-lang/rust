#![deny(unused_imports)]

use std::io::BufRead; //~ ERROR unused import: `std::io::BufRead`

fn a() {}
fn b() {}

mod test {
    use super::a; //~ ERROR unused import: `super::a`

    #[test]
    fn foo() {
        a();
        use crate::b;
    }
}

mod tests {
    use super::a; //~ ERROR unused import: `super::a`

    #[test]
    fn foo() {
        a();
        use crate::b;
    }
}

mod test_a {
    use super::a; //~ ERROR unused import: `super::a`

    #[test]
    fn foo() {
        a();
        use crate::b;
    }
}

mod a_test {
    use super::a; //~ ERROR unused import: `super::a`

    #[test]
    fn foo() {
        a();
        use crate::b;
    }
}

mod tests_a {
    use super::a; //~ ERROR unused import: `super::a`

    #[test]
    fn foo() {
        a();
        use crate::b;
    }
}

mod a_tests {
    use super::a; //~ ERROR unused import: `super::a`

    #[test]
    fn foo() {
        a();
        use crate::b;
    }
}

mod fastest_search {
    use super::a; //~ ERROR unused import: `super::a`

    #[test]
    fn foo() {
        a();
        use crate::b;
    }
}

#[cfg(test)]
mod test_has_attr {
    use super::a;

    #[test]
    fn foo() {
        a();
        use crate::b;
    }
}

fn main() {}
