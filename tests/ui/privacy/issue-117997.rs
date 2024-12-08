//@ aux-build:issue-117997.rs
//@ build-pass

extern crate issue_117997;

pub fn main() {
    issue_117997::get_assoc().method_on_assoc();
}
