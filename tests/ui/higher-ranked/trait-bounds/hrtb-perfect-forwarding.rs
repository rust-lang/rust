// Test a case where you have an impl of `Foo<X>` for all `X` that
// is being applied to `for<'a> Foo<&'a mut X>`. Issue #19730.

trait Foo<X> {
    fn foo(&mut self, x: X) {}
}

trait Bar<X> {
    fn bar(&mut self, x: X) {}
}

impl<'a, X, F> Foo<X> for &'a mut F where F: Foo<X> + Bar<X> {}

impl<'a, X, F> Bar<X> for &'a mut F where F: Bar<X> {}

fn no_hrtb<'b, T>(mut t: T) //~ WARN function cannot return
where
    T: Bar<&'b isize>,
{
    // OK -- `T : Bar<&'b isize>`, and thus the impl above ensures that
    // `&mut T : Bar<&'b isize>`.
    no_hrtb(&mut t);
}

fn bar_hrtb<T>(mut t: T) //~ WARN function cannot return
where
    T: for<'b> Bar<&'b isize>,
{
    // OK -- `T : for<'b> Bar<&'b isize>`, and thus the impl above
    // ensures that `&mut T : for<'b> Bar<&'b isize>`.  This is an
    // example of a "perfect forwarding" impl.
    bar_hrtb(&mut t);
}

fn foo_hrtb_bar_not<'b, T>(mut t: T) //~ WARN function cannot return
where
    T: for<'a> Foo<&'a isize> + Bar<&'b isize>,
{
    // Not OK -- The forwarding impl for `Foo` requires that `Bar` also
    // be implemented. Thus to satisfy `&mut T : for<'a> Foo<&'a
    // isize>`, we require `T : for<'a> Bar<&'a isize>`, but the where
    // clause only specifies `T : Bar<&'b isize>`.
    foo_hrtb_bar_not(&mut t);
    //~^ ERROR implementation of `Bar` is not general enough
    //~^^ ERROR lifetime may not live long enough
}

fn foo_hrtb_bar_hrtb<T>(mut t: T) //~ WARN function cannot return
where
    T: for<'a> Foo<&'a isize> + for<'b> Bar<&'b isize>,
{
    // OK -- now we have `T : for<'b> Bar<&'b isize>`.
    foo_hrtb_bar_hrtb(&mut t);
}

fn main() {}
