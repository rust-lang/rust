use rustc_data_structures::sync::{RwLock, ReadGuard, MappedReadGuard};

/// The `Steal` struct is intended to used as the value for a query.
/// Specifically, we sometimes have queries (*cough* MIR *cough*)
/// where we create a large, complex value that we want to iteratively
/// update (e.g., optimize). We could clone the value for each
/// optimization, but that'd be expensive. And yet we don't just want
/// to mutate it in place, because that would spoil the idea that
/// queries are these pure functions that produce an immutable value
/// (since if you did the query twice, you could observe the mutations).
/// So instead we have the query produce a `&'tcx Steal<mir::Body<'tcx>>`
/// (to be very specific). Now we can read from this
/// as much as we want (using `borrow()`), but you can also
/// `steal()`. Once you steal, any further attempt to read will panic.
/// Therefore, we know that -- assuming no ICE -- nobody is observing
/// the fact that the MIR was updated.
///
/// Obviously, whenever you have a query that yields a `Steal` value,
/// you must treat it with caution, and make sure that you know that
/// -- once the value is stolen -- it will never be read from again.
//
// FIXME(#41710): what is the best way to model linear queries?
pub struct Steal<T> {
    value: RwLock<Option<T>>
}

impl<T> Steal<T> {
    pub fn new(value: T) -> Self {
        Steal {
            value: RwLock::new(Some(value))
        }
    }

    pub fn borrow(&self) -> MappedReadGuard<'_, T> {
        ReadGuard::map(self.value.borrow(), |opt| match *opt {
            None => bug!("attempted to read from stolen value"),
            Some(ref v) => v
        })
    }

    pub fn steal(&self) -> T {
        let value_ref = &mut *self.value.try_write().expect("stealing value which is locked");
        let value = value_ref.take();
        value.expect("attempt to read from stolen value")
    }
}
