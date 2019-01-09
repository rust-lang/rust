// run-pass
// aux-build:issue_2723_a.rs

extern crate issue_2723_a;
use issue_2723_a::f;

pub fn main() {
    unsafe {
        f(vec![2]);
    }
}
