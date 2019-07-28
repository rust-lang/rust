// run-pass
// aux-build:macro_with_super_1.rs

// pretty-expanded FIXME #23616

#[macro_use]
extern crate macro_with_super_1;

declare!();

fn main() {
    bbb::ccc();
}
