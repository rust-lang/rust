use crate::alloc::Global;
use crate::co_alloc::CoAllocPref;
use core::mem::ManuallyDrop;
use core::ptr::{self};

use super::{IntoIter, SpecExtend, SpecFromIterNested, Vec};

/// Specialization trait used for Vec::from_iter
///
/// ## The delegation graph:
///
/// ```text
/// +-------------+
/// |FromIterator |
/// +-+-----------+
///   |
///   v
/// +-+-------------------------------+  +---------------------+
/// |SpecFromIter                  +---->+SpecFromIterNested   |
/// |where I:                      |  |  |where I:             |
/// |  Iterator (default)----------+  |  |  Iterator (default) |
/// |  vec::IntoIter               |  |  |  TrustedLen         |
/// |  SourceIterMarker---fallback-+  |  +---------------------+
/// +---------------------------------+
/// ```
pub(super) trait SpecFromIter<T, I> {
    fn from_iter(iter: I) -> Self;
}

#[allow(unused_braces)]
impl<T, I, const CO_ALLOC_PREF: CoAllocPref> SpecFromIter<T, I> for Vec<T, Global, CO_ALLOC_PREF>
where
    I: Iterator<Item = T>,
    [(); { crate::meta_num_slots_global!(CO_ALLOC_PREF) }]:,
{
    default fn from_iter(iterator: I) -> Self {
        SpecFromIterNested::from_iter(iterator)
    }
}

#[allow(unused_braces)]
impl<T, const CO_ALLOC_PREF: CoAllocPref> SpecFromIter<T, IntoIter<T>>
    for Vec<T, Global, CO_ALLOC_PREF>
where
    [(); { crate::meta_num_slots_global!(CO_ALLOC_PREF) }]:,
{
    fn from_iter(iterator: IntoIter<T>) -> Self {
        // A common case is passing a vector into a function which immediately
        // re-collects into a vector. We can short circuit this if the IntoIter
        // has not been advanced at all.
        // When it has been advanced We can also reuse the memory and move the data to the front.
        // But we only do so when the resulting Vec wouldn't have more unused capacity
        // than creating it through the generic FromIterator implementation would. That limitation
        // is not strictly necessary as Vec's allocation behavior is intentionally unspecified.
        // But it is a conservative choice.
        let has_advanced = iterator.buf.as_ptr() as *const _ != iterator.ptr;
        if !has_advanced || iterator.len() >= iterator.cap / 2 {
            unsafe {
                let it = ManuallyDrop::new(iterator);
                if has_advanced {
                    ptr::copy(it.ptr, it.buf.as_ptr(), it.len());
                }
                return Vec::from_raw_parts_co(it.buf.as_ptr(), it.len(), it.cap);
            }
        }

        let mut vec = Vec::<T, Global, CO_ALLOC_PREF>::new_co();
        // must delegate to spec_extend() since extend() itself delegates
        // to spec_from for empty Vecs
        vec.spec_extend(iterator);
        vec
    }
}
