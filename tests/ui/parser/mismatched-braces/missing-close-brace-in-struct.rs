pub(crate) struct Bar<T> {
  foo: T,

trait T {
    fn foo(&self);
}


impl T for Bar<usize> {
fn foo(&self) {}
}

fn main() {} //~ ERROR this file contains an unclosed delimiter
