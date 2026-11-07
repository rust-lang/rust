//@ check-pass
//@ ignore-compare-mode-polonius (explicit revisions)
//@ revisions: nll polonius
//@ [nll] compile-flags: -Zpolonius=off
//@ [polonius] compile-flags: -Zpolonius=next

// When visiting recursive aliases for liveness, we need to ignore bound
// regions (as we do in the non-recursive case).

trait Cyclic {
    type A<'a, 'b>: for<'c> PartialEq<Self::A<'a, 'c>>;
}

fn make<'a, 'b, T: Cyclic>(_: &'a String, _: &'b String) -> T::A<'a, 'b> {
    loop {}
}

fn check<T: Cyclic>() {
    let s = String::new();
    let val = make::<T>(&s, &s);
    drop(val);
}

fn main() {}
