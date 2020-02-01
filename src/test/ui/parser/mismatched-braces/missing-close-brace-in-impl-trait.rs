fn main() {}

impl T for () { //~ ERROR cannot find trait `T` in this scope

fn foo(&self) {}
//~^ ERROR missing `fn`, `type`, or `const`

trait T {
    fn foo(&self);
}

pub(crate) struct Bar<T>();

//~ ERROR this file contains an unclosed delimiter
