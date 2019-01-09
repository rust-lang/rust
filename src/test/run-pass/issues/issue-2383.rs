// run-pass
// pretty-expanded FIXME #23616

use std::collections::VecDeque;

pub fn main() {
    let mut q = VecDeque::new();
    q.push_front(10);
}
