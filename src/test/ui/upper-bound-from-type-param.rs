// Regression test found by crater run. The scenario here is that we have
// a region variable `?0` from the `impl Future` with no upper bounds
// and a member constraint of `?0 in ['a, 'static]`. If we pick `'static`,
// however, we then fail the check that `F: ?0` (since we only know that
// F: ?a). The problem here is that our upper bound detection
// doesn't consider "type tests" like `F: 'x`. This was fixed by
// preferring to pick a least choice and only using static as a last resort.
//
// edition:2018
// check-pass

trait Future {}
impl Future for () {}

fn sink_error1<'a, F: 'a>(f: F) -> impl Future + 'a {
    sink_error2(f) // error: `F` may not live long enough
}
fn sink_error2<'a, F: 'a>(_: F) -> impl Future + 'a {}
fn main() {}
