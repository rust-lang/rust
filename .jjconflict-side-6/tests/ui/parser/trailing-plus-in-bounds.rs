//@ check-pass

use std::fmt::Debug;

fn main() {
    let x: Box<dyn Debug+> = Box::new(3) as Box<dyn Debug+>; // Trailing `+` is OK
}
