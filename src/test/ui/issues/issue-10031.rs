// run-pass
// aux-build:issue-10031-aux.rs
// pretty-expanded FIXME #23616

extern crate issue_10031_aux;

pub fn main() {
    let _ = issue_10031_aux::Wrap(());
}
