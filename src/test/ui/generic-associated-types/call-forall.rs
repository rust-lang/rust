#![feature(generic_associated_types)]

trait LendingIterator {
    type Item<'a>
    where
        Self: 'a;
}

impl<'slice> LendingIterator for &'slice [u32] {
    type Item<'a> = &'a [u32]
    where
        Self: 'a;
}

impl<'slice> LendingIterator for [u32] {
    type Item<'a> = &'a [u32]
    where
        Self: 'a;
}

fn call_me<T: ?Sized>()
where
    for<'a> T: LendingIterator<Item<'a> = &'a [u32]>,
{
    if false {
        call_me::<T>();
    }
}

fn ok_i_will1<'test>() {
    // Gets an error because we cannot prove that, for all 'a, `&'test [u32]: 'a`.
    //
    // This is a bug -- what should happen is that there is an implied bound
    // so that `for<'a>` really means "for all `'a` that make sense", in which case
    // this ought to be provable.
    call_me::<&'test [u32]>; //~ ERROR lifetime may not live long enough
}

fn ok_i_will2() {
    // OK because, for all 'a, `[u32]: 'a`.
    call_me::<[u32]>;
}
fn main() {}
