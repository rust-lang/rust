pub(crate) struct Bar<T> { //~ ERROR `main` function not found
  foo: T,

trait T { //~ ERROR expected identifier, found keyword `trait`
//~^ ERROR expected `:`, found `T`
    fn foo(&self);
}


impl T for Bar<usize> {
fn foo(&self) {}
}

fn main() {} //~ ERROR this file contains an un-closed delimiter
