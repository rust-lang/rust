fn f() where u8 = u16 {}
//~^ ERROR equality constraints are not yet supported in `where` clauses
//~| ERROR mismatched types
fn g() where for<'a> &'static (u8,) = u16, {}
//~^ ERROR equality constraints are not yet supported in `where` clauses
//~| ERROR mismatched types

fn main() {}
