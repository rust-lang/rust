//@ revisions: current next
//@[next] compile-flags: -Znext-solver
//@ ignore-compare-mode-next-solver (explicit revisions)
//@[next] check-pass

// A test which shows that autoderef can constrain opaque types even
// though it's supposed to treat not-yet-defined opaque types as
// mostly rigid. I don't think this should necessarily compile :shrug:
use std::ops::Deref;

struct Wrapper<T>(T);

impl<T> Deref for Wrapper<Vec<T>> {
    type Target = Vec<T>;
    fn deref(&self) -> &Self::Target {
        &self.0
    }
}

fn foo() -> impl Sized {
    if false {
        let _ = Wrapper(foo()).len();
        //[current]~^ ERROR no method named `len` found for struct `Wrapper<T>` in the current scope
    }

    std::iter::once(1).collect()
}
fn main() {}
