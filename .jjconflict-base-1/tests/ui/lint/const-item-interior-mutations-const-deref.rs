// Regression test for <https://github.com/rust-lang/rust/issues/150157>
//
// We shouldn't lint on user types, including through deref.

//@ check-pass

use std::cell::Cell;
use std::ops::Deref;

// Cut down version of the issue reproducer without the thread local to just a Deref
pub struct LocalKey<T> {
    inner: T,
}

impl<T> Deref for LocalKey<T> {
    type Target = T;

    fn deref(&self) -> &Self::Target {
        &self.inner
    }
}

const LOCAL_COUNT: LocalKey<Cell<usize>> = LocalKey { inner: Cell::new(8) };

fn main() {
    let count = LOCAL_COUNT.get();
    LOCAL_COUNT.set(count);
}
