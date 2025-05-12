// Test that we consider `for<'a> T: 'a` to be sufficient to prove
// that `for<'a> T: 'a`.
//
//@ check-pass

#![allow(warnings)]

fn self_wf1<T>()
where
    for<'a> T: 'a,
{
    self_wf1::<T>();
}

fn main() {}
