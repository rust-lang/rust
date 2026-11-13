// Test that we consider `for<'a> &'a T: 'a` to be sufficient to prove
// that `for<'a> &'a T: 'a`.
//
// FIXME. Except we don't!

#![allow(warnings)]

fn self_wf2<T>()
where
    for<'a> &'a T: 'a,
{
    self_wf2::<T>();
    //~^ ERROR `T` does not live long enough
    //
    // FIXME. This ought to be accepted, presumably.
}

fn main() {}
