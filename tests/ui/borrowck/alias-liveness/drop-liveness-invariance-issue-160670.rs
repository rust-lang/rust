// From https://github.com/rust-lang/rust/issues/160670 This issue was
// discovered when developing Polonius alpha. A live local (the one holding the
// struct `D`) had its type be drop-live, but only partially. This had not
// previously triggered any issues because it did not affect region liveness,
// but it did affect Polonius' region variance computations, since the outer `D`
// nesting was removed to obtain `fn(&'a T)`, which unlike the associated type
// isn't invariant.
//
// The split declaration/assignment on lines 30--31 is load bearing; without
// them the bug does not appear.

struct D<T: HasArg>(T::Arg);

trait HasArg {
    type Arg;
}
impl<'a, T> HasArg for fn(&'a T) {
    type Arg = &'a T;
}
impl<T: HasArg> Drop for D<T> {
    fn drop(&mut self) {}
}
fn mk<'a, T>(r: &'a T) -> D<fn(&'a T)> {
    D(r)
}

fn main() {
    let b = Box::new(0u8);
    let d;
    d = mk(&*b);
    drop(b); //~ ERROR cannot move out of `b` because it is borrowed
}
