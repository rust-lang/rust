// Regression test for #104005.
//
// Previously, different borrowck implementations used to disagree here.
// The status of each is documented on `fn test_*`.

//@ check-fail

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

// AST: fail
// HIR: pass
// MIR: pass
pub fn test_call<'a>(val: &'a str) {
    extend_lt(val);
    //~^ ERROR borrowed data escapes outside of function
}

// AST: fail
// HIR: fail
// MIR: pass
pub fn test_coercion<'a>() {
    let _: fn(&'a str) -> _ = extend_lt;
    //~^ ERROR lifetime may not live long enough
}

// AST: fail
// HIR: fail
// MIR: fail
pub fn test_arg() {
    fn want<I, O>(_: I, _: impl Fn(I) -> O) {}
    want(&String::new(), extend_lt);
    //~^ ERROR temporary value dropped while borrowed
}

// An exploit of the unsoundness.
fn main() {
    let val = extend_lt(&String::from("blah blah blah"));
    //~^ ERROR temporary value dropped while borrowed
    println!("{}", val);
}
