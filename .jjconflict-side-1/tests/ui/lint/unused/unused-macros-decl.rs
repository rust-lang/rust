#![feature(decl_macro)]
#![deny(unused_macros)]
// To make sure we are not hitting this
#![deny(unused_macro_rules)]

// Most simple case
macro unused { //~ ERROR: unused macro definition
    () => {}
}

#[allow(unused_macros)]
mod bar {
    // Test that putting the #[deny] close to the macro's definition
    // works.

    #[deny(unused_macros)]
    macro unused { //~ ERROR: unused macro definition
        () => {}
    }
}

mod boo {
    pub(crate) macro unused { //~ ERROR: unused macro definition
        () => {}
    }
}

fn main() {}
