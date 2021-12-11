// Test that we don't hit the recursion limit for short cycles involving lifetimes.

// Shouldn't hit this, we should realize that we're in a cycle sooner.
#![recursion_limit="20"]

trait NotAuto {}
trait Y {
    type P;
}

impl<'a> Y for C<'a> {
    type P = Box<X<C<'a>>>;
}

struct C<'a>(&'a ());
struct X<T: Y>(T::P);

impl<T: NotAuto> NotAuto for Box<T> {} //~ NOTE: required
impl<T: Y> NotAuto for X<T> where T::P: NotAuto {}
impl<'a> NotAuto for C<'a> {}

fn is_send<S: NotAuto>() {}
//~^ NOTE: required
//~| NOTE: required

fn main() {
    // Should only be a few notes.
    is_send::<X<C<'static>>>();
    //~^ ERROR overflow evaluating
    //~| 3 redundant requirements hidden
    //~| required because of
}
