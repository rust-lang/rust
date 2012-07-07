// error-pattern:unresolved import
import spam::{ham, eggs};

mod spam {
    fn ham() { }
}

fn main() { ham(); eggs(); }
