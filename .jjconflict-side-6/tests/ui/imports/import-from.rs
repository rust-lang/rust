//@ run-pass

use spam::{ham, eggs};

mod spam {
    pub fn ham() { }
    pub fn eggs() { }
}

pub fn main() { ham(); eggs(); }
