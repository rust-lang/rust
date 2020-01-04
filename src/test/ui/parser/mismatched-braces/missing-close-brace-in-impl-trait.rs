impl T for () { //~ ERROR cannot find trait `T` in this scope

fn foo(&self) {}

trait T { //~ ERROR expected one of
    fn foo(&self);
}

pub(crate) struct Bar<T>();

fn main() {}
//~ ERROR this file contains an unclosed delimiter
