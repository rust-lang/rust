// Regression test for #59311. The test is taken from
// rust-lang/rust/issues/71546#issuecomment-620638437
// as they seem to have the same cause.

// FIXME: It's not clear that this code ought to report
// an error, but the regression test is here to ensure
// that it does not ICE. See discussion on #74889 for details.

pub trait T {
    fn t<F: Fn()>(&self, _: F) {}
}

pub fn crash<V>(v: &V)
where
    for<'a> &'a V: T + 'static,
{
    v.t(|| {});
    //~^ ERROR: higher-ranked lifetime error
    //~| ERROR: higher-ranked lifetime error
}

fn main() {}
