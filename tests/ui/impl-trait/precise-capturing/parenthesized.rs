// Ensure that we forbid parenthesized use-bounds. In the future we might want
// to lift this restriction but for now they bear no use whatsoever.

fn f() -> impl Sized + (use<>) {}
//~^ ERROR precise capturing lists may not be parenthesized
//~| HELP remove the parentheses

fn main() {}
