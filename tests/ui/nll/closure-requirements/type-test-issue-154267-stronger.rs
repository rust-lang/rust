// Stronger version of `type-test-issue-154267` where `error` should fail because it does not have
// an explicit `'a: 'b` bound.

struct Arg<'a: 'b, 'b: 'd, 'c: 'd, 'd, T> {
    field: *mut (&'a (), &'b (), &'c (), &'d (), T),
}
impl<'a: 'b, 'b, 'c, 'd, T> Arg<'a, 'b, 'c, 'd, T> {
    fn constrain(self)
    where
        T: 'a,
        T: 'd,
    {
    }
}
fn takes_closure<'a: 'b, 'b, 'c, T>(_: impl for<'d> FnOnce(Arg<'a, 'b, 'c, 'd, T>)) {}

fn error<'a, 'b, 'c, T: 'a>() {
    takes_closure::<'a, 'b, 'c, T>(|arg| arg.constrain()); //~ ERROR: lifetime may not live long enough
}
fn main() {}
