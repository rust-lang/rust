//@ check-pass
//
// Check that the compiler correctly detects that we don't need to propagate `T: 'b` and `T: 'c`
// when doing the type tests of `constrain`. This is a stronger version of issue-154267.

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
fn takes_closure<'a, 'b, 'c, T>(_: impl for<'d> FnOnce(Arg<'a, 'b, 'c, 'd, T>)) {}

fn error<'a, 'b, 'c, T: 'a>() {
    takes_closure::<'a, 'b, 'c, T>(|arg| arg.constrain());
}
fn main() {}
