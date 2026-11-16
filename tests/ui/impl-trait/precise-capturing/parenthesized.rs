// Ensure that we forbid parenthesized use-bounds. In the future we might want
// to lift this restriction but for now they bear no use whatsoever.

fn f() -> impl Sized + (use<>) {}
//~^ ERROR precise capturing lists may not be parenthesized
//~| HELP remove the parentheses

#[cfg(false)]
type O = Trait + (use<>);
//~^ ERROR precise capturing lists may not be parenthesized
//~| HELP remove the parentheses

// We once used to accidentally accept this.
#[cfg(false)]
type O = (use<>) + Trait;
//~^ ERROR expected a path on the left-hand side of `+`

fn main() {}
