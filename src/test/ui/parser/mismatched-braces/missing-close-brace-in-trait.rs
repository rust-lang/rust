trait T {
    fn foo(&self);

pub(crate) struct Bar<T>();
//~^ ERROR struct not supported in `trait` or `impl`

impl T for Bar<usize> {
//~^ ERROR implementation not supported in `trait` or `impl`
fn foo(&self) {}
}

fn main() {} //~ ERROR this file contains an unclosed delimiter
