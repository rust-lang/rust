// run-pass
#![allow(unreachable_code)]
// pretty-expanded FIXME #23616

fn main() {
    ({return},);
}
