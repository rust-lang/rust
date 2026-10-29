//@ revisions: valid short_x short_y short_both
//@[valid] check-pass
//@[short_x] check-fail
//@[short_y] check-fail
//@[short_both] check-fail
//@ compile-flags: -Znext-solver=globally -Zpolonius=off
//@ dont-check-compiler-stderr

// different types can have the same lifetime requirements

trait NeedsRegions<'a> {}

impl<'a, 'x, 'y> NeedsRegions<'a> for (&'x str, &'y str)
where
    (&'x str, u8): 'a,
    (&'x str, u16): 'a,
    (&'y str, u8): 'a,
    (&'y str, u16): 'a,
    (&'x str, &'y str, u8): 'a,
    (&'x str, &'y str, u16): 'a,
{}

trait Check {}

impl<'x, 'y> Check for (&'x str, &'y str)
where
    for<'a> (&'x str, &'y str): NeedsRegions<'a>,
{}

fn require<T: Check>() {}

#[cfg(valid)]
fn check() {
    require::<(&'static str, &'static str)>();
}

#[cfg(short_x)]
fn check<'short>(_: &'short str) {
    require::<(&'short str, &'static str)>();
    //[short_x]~^ ERROR lifetime may not live long enough
}

#[cfg(short_y)]
fn check<'short>(_: &'short str) {
    require::<(&'static str, &'short str)>();
    //[short_y]~^ ERROR lifetime may not live long enough
}

#[cfg(short_both)]
fn check<'short>(_: &'short str) {
    require::<(&'short str, &'short str)>();
    //[short_both]~^ ERROR lifetime may not live long enough
}

fn main() {}
