// Check that even though Cell: DispatchFromDyn it remains an invalid self parameter type

use std::cell::Cell;

trait Trait{
    fn cell(self: Cell<&Self>); //~ ERROR invalid `self` parameter type: `Cell<&Self>`
}

fn main() {}
