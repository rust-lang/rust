use super::{IntoIter, VecDeque};

impl<T> VecDeque<T> {
    pub(super) fn spec_from_iter<I: Iterator<Item = T>>(iterator: I) -> Self {
        let ty = core::mem::type_info::of::<I>();
        // Specialization for `I = crate::vec::IntoIter<T>`
        #[cfg(not(test))]
        if ty == core::mem::type_info::of::<crate::vec::IntoIter<T>>() {
            let iterator: crate::vec::IntoIter<T> =
                // SAFETY: we already know that `I` is `crate::vec::IntoIter<T>` by reflection
                unsafe { core::intrinsics::transmute_unchecked(iterator) };
            return iterator.into_vecdeque();
        }
        // Specialization for `I = IntoIter<T>`
        if ty == core::mem::type_info::of::<IntoIter<T>>() {
            // SAFETY: we already know that `I` is `IntoIter<T>` by reflection
            let iterator: IntoIter<T> = unsafe { core::intrinsics::transmute_unchecked(iterator) };
            return iterator.into_vecdeque();
        }
        // Since converting is O(1) now, just re-use the `Vec` logic for
        // anything where we can't do something extra-special for `VecDeque`,
        // especially as that could save us some monomorphization work
        // if one uses the same iterators (like slice ones) with both.
        crate::vec::Vec::from_iter(iterator).into()
    }
}
