//@ run-pass
//@ aux-build:issue-11225-3.rs


extern crate issue_11225_3;

pub fn main() {
    issue_11225_3::public_inlinable_function();
    issue_11225_3::public_inlinable_function_ufcs();
}
