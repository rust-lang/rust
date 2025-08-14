//@ run-pass
#![allow(unused_imports)]
// This should resolve fine even with the circular imports as
// they are not `pub`.

pub mod a {
    use crate::b::*;
}

pub mod b {
    use crate::a::*;
}

use a::*;

fn main() {
}
