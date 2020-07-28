// Regression test for #59311. The test is taken from
// rust-lang/rust/issues/71546#issuecomment-620638437
// as they seem to have the same cause.

pub trait T {
    fn t<F: Fn()>(&self, _: F) {}
}

pub fn crash<V>(v: &V)
where
    for<'a> &'a V: T + 'static,
{
    v.t(|| {}); //~ ERROR: higher-ranked subtype error
}

fn main() {}
