// issue: #104005

// Function type parameters with implicit type annotations were not
// checked for well-formedness, which allowed incorrect borrowing.

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
    // This used to compile
    let val = extend_lt(&String::from("blah blah blah"));
    //~^ ERROR temporary value dropped while borrowed
    println!("{}", val);
}
