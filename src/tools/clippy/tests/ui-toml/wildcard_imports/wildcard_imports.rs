//@aux-build:../../ui/auxiliary/proc_macros.rs
#![warn(clippy::wildcard_imports)]

mod prelude {
    pub const FOO: u8 = 1;
}

mod utils {
    pub const BAR: u8 = 1;
    pub fn print() {}
}

mod my_crate {
    pub mod utils {
        pub fn my_util_fn() {}
    }

    pub mod utils2 {
        pub const SOME_CONST: u32 = 1;
    }
}

pub use utils::*;
//~^ ERROR: usage of wildcard import
use my_crate::utils::*;
//~^ ERROR: usage of wildcard import
use prelude::*;
//~^ ERROR: usage of wildcard import

proc_macros::external! {
    use my_crate::utils2::*;

    static SOME_STATIC: u32 = SOME_CONST;
}

fn main() {
    let _ = FOO;
    let _ = BAR;
    print();
    my_util_fn();
}
