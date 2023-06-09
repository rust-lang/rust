// run-pass
// aux-build:issue-9155.rs

// pretty-expanded FIXME #23616

extern crate issue_9155;

struct Baz;

pub fn main() {
    issue_9155::Foo::new(Baz);
}
