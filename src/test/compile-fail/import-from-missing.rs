// error-pattern:unresolved import: eggs
import spam::{ham, eggs};

mod spam {
    fn ham() { }
}

fn main() { ham(); eggs(); }
