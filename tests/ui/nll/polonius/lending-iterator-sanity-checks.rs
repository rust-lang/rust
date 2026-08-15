// Some sanity checks for lending iterators with GATs. This is just some non-regression tests
// ensuring the polonius alpha analysis, the datalog implementation, and NLLs agree in these common
// cases of overlapping yielded items.

//@ ignore-compare-mode-polonius (explicit revisions)
//@ revisions: nll polonius legacy
//@ [polonius] compile-flags: -Z polonius=next
//@ [legacy] compile-flags: -Z polonius=legacy

trait LendingIterator {
    type Item<'a>
    where
        Self: 'a;
    fn next(&mut self) -> Option<Self::Item<'_>>;
}

fn use_live<T: LendingIterator>(t: &mut T) -> Option<(T::Item<'_>, T::Item<'_>)> {
    let Some(i) = t.next() else { return None };
    let Some(j) = t.next() else { return None };
    //~^ ERROR cannot borrow `*t` as mutable more than once at a time

    // `i` is obviously still (use-)live here, but we called `next` again to get `j`.
    Some((i, j))
}

fn drop_live<T: LendingIterator>(t: &mut T) {
    let i = t.next();

    // Now `i` is use-dead here, but we don't know if the iterator items have a `Drop` impl, so it's
    // still drop-live.
    let j = t.next();
    //~^ ERROR cannot borrow `*t` as mutable more than once at a time
}

// But we can still manually serialize the lifetimes with scopes (or preventing the destructor from
// being called), so they're not overlapping.
fn manually_non_overlapping<T: LendingIterator>(t: &mut T) {
    {
        let i = t.next();
    }

    let j = t.next(); // i is dead

    drop(j);
    let k = t.next(); // j is dead

    let k = std::mem::ManuallyDrop::new(k);
    let l = t.next(); // we told the compiler that k is not drop-live
}

// The cfg below is because there's a diagnostic ICE trying to explain the source of the error when
// using the datalog implementation. We're not fixing *that*, outside of removing the implementation
// in the future.
#[cfg(not(legacy))] // FIXME: remove this cfg when removing the datalog implementation
fn items_have_no_borrows<T: LendingIterator>(t: &mut T)
where
    for<'a> T::Item<'a>: 'static,
{
    let i = t.next();
    let j = t.next();
}

fn items_are_copy<T: LendingIterator>(t: &mut T)
where
    for<'a> T::Item<'a>: Copy,
{
    let i = t.next();
    let j = t.next();
}

fn main() {}
