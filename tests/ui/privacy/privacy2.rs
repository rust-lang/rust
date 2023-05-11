#![feature(start, no_core)]
#![no_core] // makes debugging this test *a lot* easier (during resolve)

// Test to make sure that globs don't leak in regular `use` statements.

mod bar {
    pub use self::glob::*;

    pub mod glob {
        use foo;
    }
}

pub fn foo() {}

fn test1() {
    use bar::foo;
    //~^ ERROR unresolved import `bar::foo` [E0432]
    //~| no `foo` in `bar`
}

fn test2() {
    use bar::glob::foo;
    //~^ ERROR `foo` is private
}

#[start] fn main(_: isize, _: *const *const u8) -> isize { 3 }
