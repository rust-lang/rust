//@ check-pass

#![warn(unused_imports)]

#[macro_export]
macro_rules! mac { () => {} }

fn main() {
    // Unused, `mac` as `macro_rules!` is already in scope and has higher priority.
    use crate::mac; //~ WARN unused import: `crate::mac`

    mac!();
}
