// compile-pass
// compile-flags: -Z parse-only -Z continue-parse-after-error

use std::fmt::Debug;

fn main() {
    let x: Box<Debug+> = box 3 as Box<Debug+>; // Trailing `+` is OK
}
