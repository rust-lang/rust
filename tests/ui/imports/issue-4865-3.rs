// run-pass
#![allow(unused_imports)]
// This should resolve fine even with the circular imports as
// they are not `pub`.

pub mod a {
    use b::*;
}

pub mod b {
    use a::*;
}

use a::*;

fn main() {
}
