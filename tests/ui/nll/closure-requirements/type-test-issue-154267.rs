//@ check-pass
// This test checks that the compiler doesn't propagate `T: 'b` during the `T: 'a` type test.
// If it did, it would fail to compile, even though the program is sound.

struct Arg<'a: 'c, 'b: 'c, 'c, T> {
    field: *mut (&'a (), &'b (), &'c (), T),
}

impl<'a, 'b, 'c, T> Arg<'a, 'b, 'c, T> {
    fn constrain(self)
    where
        T: 'a,
        T: 'c,
    {
    }
}

fn takes_closure<'a, 'b, T>(_: impl for<'c> FnOnce(Arg<'a, 'b, 'c, T>)) {}

// We have `'a: 'c` and `'b: 'c`, requiring `T: 'a` in `constrain` should not need
// `T: 'b` here.
fn error<'a, 'b, T: 'a>() {
    takes_closure::<'a, 'b, T>(|arg| arg.constrain());
}

fn main() {}
