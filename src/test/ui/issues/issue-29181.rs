// aux-build:issue-29181.rs

extern crate issue_29181 as foo;

fn main() {
    0.homura(); //~ ERROR no method named `homura` found
}
