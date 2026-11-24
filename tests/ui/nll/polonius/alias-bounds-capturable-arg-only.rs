//@ ignore-compare-mode-polonius (explicit revisions)
//@ edition: 2024
//@ revisions: nll polonius
//@ [nll] compile-flags: -Z polonius=off
//@ [polonius] compile-flags: -Z polonius=next

// Alias outlives obligations currently use type tests, with one way to prove
// `Alias: 'r` being that *all regions within `Alias` outlive 'r*. This is
// conservative: that also includes regions that *cannot* be in the alias,
// because of the alias outlives bounds.
//
// The test has a few cases where subsetting to only requiring *live* regions
// in `Alias` to outlive `r` can be more precise.

fn make<'a, 'b, 'c, 'z>(c: &'c u8, _z: &'z u8) -> impl Copy + 'a + 'b + use<'a, 'b, 'c, 'z>
where
    'c: 'a,
    'c: 'b,
    'z: 'z,
{
    c
}

fn require<'x, T: 'x>(_: T) {}

fn f<'x, 'a, 'b, 'c, 'z>(c: &'c u8, z: &'z u8)
where
    'c: 'x,
    'c: 'a,
    'c: 'b,
{
    require::<'x, _>(make::<'a, 'b, 'c, 'z>(c, z));
    //[nll,polonius]~^ ERROR the opaque type
}

mod relation_filter {
    trait Marker<'z> {}
    impl<'z, T> Marker<'z> for T {}

    fn make<'a, 'b, 'c, 'z>(c: &'c u8, _z: &'z u8) -> impl Copy + 'a + 'b + Marker<'z>
    where
        'c: 'a,
        'c: 'b,
        'z: 'z,
    {
        c
    }

    fn require_static<T: 'static>(_: T) {}

    fn test() {
        let z = 0u8;
        let value = make::<'static, 'static, 'static, '_>(&0, &z);
        require_static(value);
    }
}

fn main() {}
