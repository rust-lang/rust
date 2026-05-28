// Regression test for <https://github.com/rust-lang/rust/issues/144165>.

// We were previously suppressing the copy error in the `Clone` impl because we assumed
// that the only way we get `Copy` ambiguity errors was due to incoherent impls. This is
// not true, since ambiguities can be encountered due to overflows (among other ways).

struct S<T: 'static>(Option<&'static T>);

impl<T: 'static> Copy for S<T> where S<T>: Copy + Clone {}
impl<T: 'static> Clone for S<T> {
    fn clone(&self) -> Self {
        *self
        //~^ ERROR cannot move out of `*self` which is behind a shared reference
    }
}
fn main() {}
