//@ revisions: edition2015 edition2024 polonius_alpha
//@ ignore-compare-mode-polonius (explicit revisions)
//@ [edition2015] edition: 2015
//@ [edition2024] edition: 2024
//@ [polonius_alpha] edition: 2024
//@ [polonius_alpha] compile-flags: -Zpolonius=next

trait Updater {
    type Changes<'a, 'b: 'a>
    where
        Self: 'a;
    fn changes<'a, 'b>(&'a mut self, _value: &'b u8) -> Self::Changes<'a, 'b>;
}

fn overlapping_mut<T>(mut t: T)
where
    T: Updater,
    for<'a, 'b> T::Changes<'a, 'b>: 'static,
{
    let static_unit: &'static u8 = &0;
    let a = t.changes(static_unit);
    let b = t.changes(static_unit);
}

fn overlapping_mut2<T>(mut t: T)
where
    T: Updater,
    for<'a, 'b> T::Changes<'a, 'b>: 'b,
{
    let static_unit: &'static u8 = &0;
    let a = t.changes(static_unit);
    let b = t.changes(static_unit);
    // ^ These don't error, because the underlying type can't capture `&self`:
    // it would have to outlive `'b` for arbitrary `'a` and `'b`.
}

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
