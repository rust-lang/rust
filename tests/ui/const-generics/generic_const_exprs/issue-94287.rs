// aux-build:issue-94287-aux.rs
// build-fail

extern crate issue_94287_aux;

use std::str::FromStr;

fn main() {
    let _ = <issue_94287_aux::FixedI32<16>>::from_str("");
}
