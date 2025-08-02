// FIXME(estebank): diagnostics with long type paths that don't print out the full path anywhere
// still prints the note explaining where the type was written to.
//@ compile-flags: -Zwrite-long-types-to-disk=yes
trait Output<'a> {
    type Type;
}

struct Wrapper;

impl Wrapper {
    fn do_something_wrapper<O, F>(self, _: F)
    //~^ ERROR the trait bound `for<'a> F: Output<'a>` is not satisfied
    where
        F: for<'a> FnOnce(<F as Output<'a>>::Type),
        //~^ ERROR the trait bound `for<'a> F: Output<'a>` is not satisfied
        //~| ERROR the trait bound `for<'a> F: Output<'a>` is not satisfied
    {
    }
}

fn main() {
    let mut wrapper = Wrapper;
    wrapper.do_something_wrapper(|value| ());
    //~^ ERROR expected a `FnOnce
}
