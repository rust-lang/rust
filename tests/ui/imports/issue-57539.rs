//@ edition:2018

mod core {
    use core; //~ ERROR `core` is ambiguous
    use crate::*;
}

fn main() {}
