// Regression test for ICE #124848
// Tests that there is no ICE when a cast
// involves a type with error

use std::cell::Cell;

struct MyType<'a>(Cell<Option<&'unpinned mut MyType<'a>>>, Pin);
//~^ ERROR use of undeclared lifetime name `'unpinned`
//~| ERROR cannot find type `Pin` in this scope

fn main() {
    let mut unpinned = MyType(Cell::new(None));
    //~^ ERROR his struct takes 2 arguments but 1 argument was supplied
    let bad_addr = &unpinned as *const Cell<Option<&'a mut MyType<'a>>> as usize;
    //~^ ERROR use of undeclared lifetime name `'a`
    //~| ERROR use of undeclared lifetime name `'a`
    //~| ERROR casting `&MyType<'_>` as `*const Cell<Option<&mut MyType<'_>>>` is invalid
}
