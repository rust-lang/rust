#![allow(unused_imports)]

mod inner {
    pub enum Example {
        ExOne,
    }
}

mod reexports {
    pub use crate::inner::Example as _;
}

use crate::reexports::*;
//~^ SUGGESTION: use inner::Example::ExOne

fn main() {
    ExOne;
    //~^ ERROR: cannot find value `ExOne` in this scope
}
