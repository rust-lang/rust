// run-pass
// aux-build:issue-9188.rs


extern crate issue_9188;

pub fn main() {
    let a = issue_9188::bar();
    let b = issue_9188::foo::<isize>();
    assert_eq!(*a, *b);
}
