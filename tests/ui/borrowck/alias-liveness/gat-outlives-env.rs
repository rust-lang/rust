//@ revisions: edition2015 edition2024 polonius_alpha
//@ ignore-compare-mode-polonius (explicit revisions)
//@ [edition2015] edition: 2015
//@ [edition2024] edition: 2024
//@ [polonius_alpha] edition: 2024
//@ [polonius_alpha] compile-flags: -Zpolonius=next

// Demonstrates that you can use a outlives bound on an associated type in a
// where clause to restrict the lifetimes that may be contained within.

trait Updater {
    type Changes<'a, 'b: 'a>
    where
        Self: 'a;
    fn changes<'a, 'b>(&'a mut self, _value: &'b u8) -> Self::Changes<'a, 'b>;
}

// Nothing can be captured.
fn overlapping_mut<T>(mut t: T)
where
    T: Updater,
    for<'a, 'b> T::Changes<'a, 'b>: 'static,
{
    let static_unit: &'static u8 = &0;
    let a = t.changes(static_unit);
    let b = t.changes(static_unit);
}

// Only `&mut self` is captured not `&'static u8`, and that borrow is killed on the next call.
fn overlapping_mut2<T>(mut t: T)
where
    T: Updater,
    for<'a, 'b> T::Changes<'a, 'b>: 'b,
{
    let static_unit: &'static u8 = &0;
    let a = t.changes(static_unit);
    let b = t.changes(static_unit);
}

// Both `&mut self` and `&'static u8` are captured, because the `'static` lifetime
// makes the associated type live for the entire function body.
fn overlapping_mut3<'b, T>(mut t: T)
where
    T: Updater,
    for<'a> T::Changes<'a, 'b>: 'a,
{
    let static_unit: &'static u8 = &0;
    let a = t.changes(static_unit);
    let b = t.changes(static_unit); //[edition2015,edition2024,polonius_alpha]~ ERROR cannot borrow `t` as mutable more than once at a time
}

fn main() {}
