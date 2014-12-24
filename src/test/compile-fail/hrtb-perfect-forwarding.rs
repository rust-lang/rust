// Copyright 2014 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

// Test a case where you have an impl of `Foo<X>` for all `X` that
// is being applied to `for<'a> Foo<&'a mut X>`. Issue #19730.

trait Foo<X> {
    fn foo(&mut self, x: X) { }
}

trait Bar<X> {
    fn bar(&mut self, x: X) { }
}

impl<'a,X,F> Foo<X> for &'a mut F
    where F : Foo<X> + Bar<X>
{
}

impl<'a,X,F> Bar<X> for &'a mut F
    where F : Bar<X>
{
}

fn no_hrtb<'b,T>(mut t: T)
    where T : Bar<&'b int>
{
    // OK -- `T : Bar<&'b int>`, and thus the impl above ensures that
    // `&mut T : Bar<&'b int>`.
    no_hrtb(&mut t);
}

fn bar_hrtb<T>(mut t: T)
    where T : for<'b> Bar<&'b int>
{
    // OK -- `T : for<'b> Bar<&'b int>`, and thus the impl above
    // ensures that `&mut T : for<'b> Bar<&'b int>`.  This is an
    // example of a "perfect forwarding" impl.
    bar_hrtb(&mut t);
}

fn foo_hrtb_bar_not<'b,T>(mut t: T)
    where T : for<'a> Foo<&'a int> + Bar<&'b int>
{
    // Not OK -- The forwarding impl for `Foo` requires that `Bar` also
    // be implemented. Thus to satisfy `&mut T : for<'a> Foo<&'a
    // int>`, we require `T : for<'a> Bar<&'a int>`, but the where
    // clause only specifies `T : Bar<&'b int>`.
    foo_hrtb_bar_not(&mut t); //~ ERROR `for<'a> Bar<&'a int>` is not implemented for the type `T`
}

fn foo_hrtb_bar_hrtb<T>(mut t: T)
    where T : for<'a> Foo<&'a int> + for<'b> Bar<&'b int>
{
    // OK -- now we have `T : for<'b> Bar&'b int>`.
    foo_hrtb_bar_hrtb(&mut t);
}

fn main() { }
