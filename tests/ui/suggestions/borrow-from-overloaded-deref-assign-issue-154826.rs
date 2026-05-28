// Regression test for #154826.

use std::rc::Rc;

struct NonCopy;

fn main() {
    let b: NonCopy;
    (b,) = *Rc::new((NonCopy,));
    //~^ ERROR cannot move out of an `Rc`
    //~| HELP destructuring assignment cannot borrow from this expression; consider using a `let` binding instead
}
