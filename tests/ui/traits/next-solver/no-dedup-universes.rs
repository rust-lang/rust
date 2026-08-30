//@ compile-flags: -Znext-solver -Zno-leak-check

//! Make sure we don't drop trivial-looking region constraints that would otherwise fail
//! leak check.

trait Trait {}
trait Other<'a, 'b> {}

struct Foo;
// We need this indirection because something direct like `for<'a> &'a (): 'b` gives us a
// TypeOutlives constraint, whereas we want to be testing how we handle RegionOutlives, and
// only `impl Other for Bar`'s where-clause can give us that.
impl<'b> Trait for Foo where for<'a> Bar: Other<'a, 'b> {}

struct Bar;
impl<'a, 'b> Other<'a, 'b> for Bar where 'a: 'b {}

fn f<T: Trait>(_: T) {}

fn main() { f(Foo); }
//~^ ERROR higher-ranked lifetime error
