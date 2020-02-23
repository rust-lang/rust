fn main() {}

impl T for () { //~ ERROR cannot find trait `T` in this scope

fn foo(&self) {}

trait T { //~ ERROR trait not supported in `trait` or `impl`
    fn foo(&self);
}

pub(crate) struct Bar<T>(); //~ ERROR struct not supported in `trait` or `impl`

//~ ERROR this file contains an unclosed delimiter
