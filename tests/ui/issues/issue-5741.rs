// run-pass
// pretty-expanded FIXME #23616
#![allow(while_true)]
#![allow(unreachable_code)]

pub fn main() {
    return;
    while true {};
}
