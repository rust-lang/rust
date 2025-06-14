//@ compile-flags: -Zdeduplicate-diagnostics=yes
//@ dont-require-annotations: NOTE

#![feature(no_core)]
#![no_core] // makes debugging this test *a lot* easier (during resolve)

// Test to make sure that globs don't leak in regular `use` statements.

mod bar {
    pub use self::glob::*;

    pub mod glob {
        use crate::foo;
    }
}

pub fn foo() {}
//~^ ERROR requires `sized` lang_item

fn test1() {
    //~^ ERROR requires `sized` lang_item
    use bar::foo;
    //~^ ERROR unresolved import `bar::foo` [E0432]
    //~| NOTE no `foo` in `bar`
}

fn test2() {
    //~^ ERROR requires `sized` lang_item
    use bar::glob::foo;
    //~^ ERROR `foo` is private
}

fn main() {}
//~^ ERROR requires `sized` lang_item
