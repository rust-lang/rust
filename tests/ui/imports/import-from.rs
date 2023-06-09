// run-pass
// pretty-expanded FIXME #23616

use spam::{ham, eggs};

mod spam {
    pub fn ham() { }
    pub fn eggs() { }
}

pub fn main() { ham(); eggs(); }
