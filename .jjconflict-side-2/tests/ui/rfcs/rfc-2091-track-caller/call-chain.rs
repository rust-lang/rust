//@ run-pass
//@ revisions: default mir-opt
//@[default] compile-flags: -Zinline-mir=false
//@[mir-opt] compile-flags: -Zmir-opt-level=4

use std::panic::Location;

struct Foo;

impl Foo {
    #[inline(always)]
    #[track_caller]
    fn check_loc(&self, line: u32, col: u32) -> &Self {
        let loc = Location::caller();
        assert_eq!(loc.file(), file!(), "file mismatch");
        assert_eq!(loc.line(), line, "line mismatch");
        assert_eq!(loc.column(), col, "column mismatch");
        self
    }
}

fn main() {
    // Tests that when `Location::caller` is used in a method chain,
    // it points to the start of the correct call (the first character after the dot)
    // instead of to the very first expression in the chain
    let foo = Foo;
    foo.
        check_loc(line!(), 9).check_loc(line!(), 31)
        .check_loc(line!(), 10);
}
