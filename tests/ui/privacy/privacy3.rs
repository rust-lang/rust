//@ compile-flags: -Zdeduplicate-diagnostics=yes

#![feature( no_core)]
#![no_core] // makes debugging this test *a lot* easier (during resolve)

// Test to make sure that private items imported through globs remain private
// when  they're used.

mod bar {
    pub use self::glob::*;

    mod glob {
        fn gpriv() {}
    //~^ ERROR requires `sized` lang_item
    }
}

pub fn foo() {}
//~^ ERROR requires `sized` lang_item

fn test1() {
    //~^ ERROR requires `sized` lang_item
    use bar::gpriv;
    //~^ ERROR unresolved import `bar::gpriv` [E0432]
    //~| NOTE no `gpriv` in `bar`

    // This should pass because the compiler will insert a fake name binding
    // for `gpriv`
    gpriv();
}

fn main() {}
//~^ ERROR requires `sized` lang_item
