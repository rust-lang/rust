// aux-build:issue-56943.rs

extern crate issue_56943;

fn main() {
    let _: issue_56943::S = issue_56943::S2;
    //~^ ERROR mismatched types [E0308]
}
