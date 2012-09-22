// error-pattern:unresolved
// xfail-test
use spam::{ham, eggs};

mod spam {
    #[legacy_exports];
    fn ham() { }
}

fn main() { ham(); eggs(); }
