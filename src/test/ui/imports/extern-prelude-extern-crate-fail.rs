// aux-build:two_macros.rs

mod n {
    extern crate two_macros;
}

mod m {
    fn check() {
        two_macros::m!(); //~ ERROR failed to resolve. Use of undeclared type or module `two_macros`
    }
}

fn main() {}
