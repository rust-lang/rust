//@ revisions: old next
//@[next] compile-flags: -Znext-solver
//@ ignore-compare-mode-next-solver (explicit revisions)
//@ check-pass

// Regression test for #158461. Outlives clauses from the parameter environment
// need to be normalized before alias liveness analysis can match them.

trait Id {
    type SelfType;
}

impl<T> Id for T {
    type SelfType = T;
}

trait Foo {
    type Assoc<'a>
    where
        Self: 'a;

    fn assoc(&mut self) -> Self::Assoc<'_>;
}

// The normalized `'static` bound allows this value's borrow to end immediately.
fn overlapping_mut<T>(mut t: T)
where
    T: Foo,
    for<'a> <T::Assoc<'a> as Id>::SelfType: 'static,
{
    let a = t.assoc();
    let b = t.assoc();
}

// This is a distinct liveness path: the owner can be moved while the projected
// value remains live.
fn live_past_borrow<T>(mut t: T)
where
    T: Foo,
    for<'a> <T::Assoc<'a> as Id>::SelfType: 'static,
{
    let x = t.assoc();
    drop(t);
    drop(x);
}

fn main() {}
