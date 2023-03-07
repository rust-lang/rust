// aux-build:issue-92755.rs
// build-pass

// Thank you @tmiasko for providing the content of this test!

extern crate issue_92755;

fn main() {
    issue_92755::ctx().a.b.f();
}
