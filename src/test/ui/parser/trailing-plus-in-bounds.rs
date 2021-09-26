// build-pass (FIXME(62277): could be check-pass?)

#![allow(bare_trait_objects)]

use std::fmt::Debug;

fn main() {
    let x: Box<Debug+> = Box::new(3) as Box<Debug+>; // Trailing `+` is OK
}
