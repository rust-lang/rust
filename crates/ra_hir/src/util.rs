//! Internal utility functions.

use std::sync::Arc;

/// Helper for mutating `Arc<[T]>` (i.e. `Arc::make_mut` for Arc slices).
/// The underlying values are cloned if there are other strong references.
pub(crate) fn make_mut_arc_slice<T: Clone, R>(
    a: &mut Arc<[T]>,
    f: impl FnOnce(&mut [T]) -> R,
) -> R {
    if let Some(s) = Arc::get_mut(a) {
        f(s)
    } else {
        let mut v = a.to_vec();
        let r = f(&mut v);
        *a = Arc::from(v);
        r
    }
}
