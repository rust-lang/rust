fn main() {}

impl T for () { //~ ERROR cannot find trait `T` in this scope

fn foo(&self) {}
//~^ ERROR missing `fn`, `type`, `const`, or `static` for item declaration

trait T {
    fn foo(&self);
}

pub(crate) struct Bar<T>();

//~ ERROR this file contains an unclosed delimiter
