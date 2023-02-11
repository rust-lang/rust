// run-pass
// aux-build:issue-11508.rs

extern crate issue_11508 as rand;

use rand::{Closed01, random};

fn main() {
    let Closed01(val) = random::<Closed01<f32>>();
    println!("{}", val);
}
