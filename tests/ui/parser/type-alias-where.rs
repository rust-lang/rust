// check-fail

// Fine, but lints as unused
type Foo where u32: Copy = ();
// Not fine.
type Bar = () where u32: Copy;
//~^ ERROR where clauses are not allowed
type Baz = () where;
//~^ ERROR where clauses are not allowed

fn main() {}
