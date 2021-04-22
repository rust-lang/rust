// aux-build:issue-84429.rs
// edition:2021
// check-pass

extern crate issue_84429;

fn main() {
    issue_84429::foo!(1 | 2);
}
