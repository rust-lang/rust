// check-pass
// known-bug: #104005

// Should fail. Function type parameters with implicit type annotations are not
// checked for well-formedness, which allows incorrect borrowing.

// In contrast, user annotations are always checked for well-formedness, and the
// commented code below is correctly rejected by the borrow checker.

use std::fmt::Display;

trait Displayable {
    fn display(self) -> Box<dyn Display>;
}

impl<T: Display> Displayable for (T, Option<&'static T>) {
    fn display(self) -> Box<dyn Display> {
        Box::new(self.0)
    }
}

fn extend_lt<T, U>(val: T) -> Box<dyn Display>
where
    (T, Option<U>): Displayable,
{
    Displayable::display((val, None))
}

fn main() {
    // *incorrectly* compiles
    let val = extend_lt(&String::from("blah blah blah"));
    println!("{}", val);

    // *correctly* fails to compile
    // let val = extend_lt::<_, &_>(&String::from("blah blah blah"));
    // println!("{}", val);
}
