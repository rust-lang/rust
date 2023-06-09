// run-pass
// aux-build:issue-11529.rs

// pretty-expanded FIXME #23616

extern crate issue_11529 as a;

fn main() {
    let one = 1;
    let _a = a::A(&one);
}
