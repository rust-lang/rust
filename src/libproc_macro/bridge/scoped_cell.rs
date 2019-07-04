//! `Cell` variant for (scoped) existential lifetimes.

use std::cell::Cell;
use std::mem;
use std::ops::{Deref, DerefMut};

/// Type lambda application, with a lifetime.
pub trait ApplyL<'a> {
    type Out;
}

/// Type lambda taking a lifetime, i.e., `Lifetime -> Type`.
pub trait LambdaL: for<'a> ApplyL<'a> {}

impl<T: for<'a> ApplyL<'a>> LambdaL for T {}

// HACK(eddyb) work around projection limitations with a newtype
// FIXME(#52812) replace with `&'a mut <T as ApplyL<'b>>::Out`
pub struct RefMutL<'a, 'b, T: LambdaL>(&'a mut <T as ApplyL<'b>>::Out);

impl<'a, 'b, T: LambdaL> Deref for RefMutL<'a, 'b, T> {
    type Target = <T as ApplyL<'b>>::Out;
    fn deref(&self) -> &Self::Target {
        self.0
    }
}

impl<'a, 'b, T: LambdaL> DerefMut for RefMutL<'a, 'b, T> {
    fn deref_mut(&mut self) -> &mut Self::Target {
        self.0
    }
}

pub struct ScopedCell<T: LambdaL>(Cell<<T as ApplyL<'static>>::Out>);

impl<T: LambdaL> ScopedCell<T> {
    pub const fn new(value: <T as ApplyL<'static>>::Out) -> Self {
        ScopedCell(Cell::new(value))
    }

    /// Sets the value in `self` to `replacement` while
    /// running `f`, which gets the old value, mutably.
    /// The old value will be restored after `f` exits, even
    /// by panic, including modifications made to it by `f`.
    pub fn replace<'a, R>(
        &self,
        replacement: <T as ApplyL<'a>>::Out,
        f: impl for<'b, 'c> FnOnce(RefMutL<'b, 'c, T>) -> R,
    ) -> R {
        /// Wrapper that ensures that the cell always gets filled
        /// (with the original state, optionally changed by `f`),
        /// even if `f` had panicked.
        struct PutBackOnDrop<'a, T: LambdaL> {
            cell: &'a ScopedCell<T>,
            value: Option<<T as ApplyL<'static>>::Out>,
        }

        impl<'a, T: LambdaL> Drop for PutBackOnDrop<'a, T> {
            fn drop(&mut self) {
                self.cell.0.set(self.value.take().unwrap());
            }
        }

        let mut put_back_on_drop = PutBackOnDrop {
            cell: self,
            value: Some(self.0.replace(unsafe {
                let erased = mem::transmute_copy(&replacement);
                mem::forget(replacement);
                erased
            })),
        };

        f(RefMutL(put_back_on_drop.value.as_mut().unwrap()))
    }

    /// Sets the value in `self` to `value` while running `f`.
    pub fn set<R>(&self, value: <T as ApplyL<'_>>::Out, f: impl FnOnce() -> R) -> R {
        self.replace(value, |_| f())
    }
}
