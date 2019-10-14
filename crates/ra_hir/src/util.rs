//! Internal utility functions.

use std::sync::Arc;

/// Helper for mutating `Arc<[T]>` (i.e. `Arc::make_mut` for Arc slices).
/// The underlying values are cloned if there are other strong references.
pub(crate) fn make_mut_slice<T: Clone>(a: &mut Arc<[T]>) -> &mut [T] {
    if Arc::get_mut(a).is_none() {
        *a = a.iter().cloned().collect();
    }
    Arc::get_mut(a).unwrap()
}
