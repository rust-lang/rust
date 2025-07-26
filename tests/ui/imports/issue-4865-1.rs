//@ run-pass
#![allow(unused_imports)]
// This should resolve fine.
// Prior to fix, the crossed imports between a and b
// would block on the glob import, itself never being resolved
// because these previous imports were not resolved.

pub mod a {
    use crate::b::fn_b;
    use crate::c::*;

    pub fn fn_a(){
    }
}

pub mod b {
    use crate::a::fn_a;
    use crate::c::*;

    pub fn fn_b(){
    }
}

pub mod c{
    pub fn fn_c(){
    }
}

use a::fn_a;
use b::fn_b;

fn main() {
}
