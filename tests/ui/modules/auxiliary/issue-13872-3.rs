extern crate issue_13872_2 as bar;

use bar::B;

pub fn foo() {
    match B {
        B => {}
    }
}
