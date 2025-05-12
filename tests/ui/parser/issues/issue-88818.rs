// Regression test for #88818 (improve error message for missing trait
// in `impl for X`).

struct S { }
impl for S { }
//~^ ERROR: missing trait in a trait impl
//~| HELP: add a trait here
//~| HELP: for an inherent impl, drop this `for`

fn main() {}
