trait T {
//~^ ERROR `main` function not found in crate `missing_close_brace_in_trait`
    fn foo(&self);

pub(crate) struct Bar<T>();
//~^ ERROR missing `fn`, `type`, `const`, or `static` for item declaration

impl T for Bar<usize> {
fn foo(&self) {}
}

fn main() {} //~ ERROR this file contains an unclosed delimiter
