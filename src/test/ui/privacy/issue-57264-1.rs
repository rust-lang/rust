// build-pass (FIXME(62277): could be check-pass?)
// aux-build:issue-57264-1.rs

extern crate issue_57264_1;

fn main() {
    issue_57264_1::Pub::pub_method();
}
