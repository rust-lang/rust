// error-pattern:unresolved
// xfail-test
use spam::{ham, eggs};

mod spam {
    fn ham() { }
}

fn main() { ham(); eggs(); }
