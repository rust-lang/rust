//@ check-fail
//@ normalize-stderr: "error: aborting due to 1 previous error\n\n" -> "error: aborting due to 1 previous error\n"

#![feature(supertrait_item_shadowing)]
#![deny(ambiguous_glob_imported_traits)]
#![allow(dead_code, unused_imports)]

trait DefaultPolicy {
    fn allow_action(&self) -> bool {
        false
    }
}

mod first_policy {
    pub trait Role: crate::DefaultPolicy {
        fn allow_action(&self) -> bool {
            true
        }
    }

    impl crate::DefaultPolicy for u8 {}
    impl Role for u8 {}
}

mod second_policy {
    pub trait Role: crate::DefaultPolicy {
        fn audit_only(&self) {}
    }
}

use first_policy::*;
use second_policy::*;

fn main() {
    assert!(0u8.allow_action());
    //~^ ERROR Use of ambiguously glob imported trait `Role`
    //~| WARN this was previously accepted by the compiler but is being phased out
}
