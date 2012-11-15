// error-pattern:failed to resolve import
use spam::{ham, eggs};

mod spam {
    #[legacy_exports];
    fn ham() { }
}

fn main() { ham(); eggs(); }
