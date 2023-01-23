//use core::alloc;
use crate::alloc::Allocator;
#[cfg(not(no_global_oom_handling))]
use crate::borrow::Cow;

use super::Vec;

macro_rules! __impl_slice_eq1 {
    ([$($vars:tt)*] $lhs:ty, $rhs:ty, #[$stability:meta], $($constraints:tt)*) => {
        #[$stability]
        impl<T, U, $($vars)*> PartialEq<$rhs> for $lhs
        where
            T: PartialEq<U>,
            $($constraints)*
        {
            #[inline]
            fn eq(&self, other: &$rhs) -> bool { self[..] == other[..] }
            #[inline]
            fn ne(&self, other: &$rhs) -> bool { self[..] != other[..] }
        }
    }
}

__impl_slice_eq1! { [A1: Allocator, A2: Allocator, const COOP_PREFERRED1: bool, const COOP_PREFERRED2: bool] Vec<T, A1, COOP_PREFERRED1>, Vec<U, A2, COOP_PREFERRED2>, #[stable(feature = "rust1", since = "1.0.0")], [(); core::alloc::co_alloc_metadata_num_slots_with_preference::<A1>(COOP_PREFERRED1)]:, [(); core::alloc::co_alloc_metadata_num_slots_with_preference::<A2>(COOP_PREFERRED2)]: }
__impl_slice_eq1! { [A: Allocator, const COOP_PREFERRED: bool] Vec<T, A, COOP_PREFERRED>, &[U], #[stable(feature = "rust1", since = "1.0.0")], [(); core::alloc::co_alloc_metadata_num_slots_with_preference::<A>(COOP_PREFERRED)]: }
__impl_slice_eq1! { [A: Allocator, const COOP_PREFERRED: bool] Vec<T, A, COOP_PREFERRED>, &mut [U], #[stable(feature = "rust1", since = "1.0.0")], [(); core::alloc::co_alloc_metadata_num_slots_with_preference::<A>(COOP_PREFERRED)]: }
__impl_slice_eq1! { [A: Allocator, const COOP_PREFERRED: bool] &[T], Vec<U, A, COOP_PREFERRED>, #[stable(feature = "partialeq_vec_for_ref_slice", since = "1.46.0")], [(); core::alloc::co_alloc_metadata_num_slots_with_preference::<A>(COOP_PREFERRED)]: }
__impl_slice_eq1! { [A: Allocator, const COOP_PREFERRED: bool] &mut [T], Vec<U, A, COOP_PREFERRED>, #[stable(feature = "partialeq_vec_for_ref_slice", since = "1.46.0")], [(); core::alloc::co_alloc_metadata_num_slots_with_preference::<A>(COOP_PREFERRED)]: }
__impl_slice_eq1! { [A: Allocator, const COOP_PREFERRED: bool] Vec<T, A, COOP_PREFERRED>, [U], #[stable(feature = "partialeq_vec_for_slice", since = "1.48.0")], [(); core::alloc::co_alloc_metadata_num_slots_with_preference::<A>(COOP_PREFERRED)]:  }
__impl_slice_eq1! { [A: Allocator, const COOP_PREFERRED: bool] [T], Vec<U, A, COOP_PREFERRED>, #[stable(feature = "partialeq_vec_for_slice", since = "1.48.0")], [(); core::alloc::co_alloc_metadata_num_slots_with_preference::<A>(COOP_PREFERRED)]:  }
#[cfg(not(no_global_oom_handling))]
__impl_slice_eq1! { [A: Allocator, const COOP_PREFERRED: bool] Cow<'_, [T]>, Vec<U, A, COOP_PREFERRED>, #[stable(feature = "rust1", since = "1.0.0")], T: Clone, [(); core::alloc::co_alloc_metadata_num_slots_with_preference::<A>(COOP_PREFERRED)]: }
#[cfg(not(no_global_oom_handling))]
__impl_slice_eq1! { [] Cow<'_, [T]>, &[U], #[stable(feature = "rust1", since = "1.0.0")], T: Clone }
#[cfg(not(no_global_oom_handling))]
__impl_slice_eq1! { [] Cow<'_, [T]>, &mut [U], #[stable(feature = "rust1", since = "1.0.0")], T: Clone }
__impl_slice_eq1! { [A: Allocator, const COOP_PREFERRED: bool, const N: usize] Vec<T, A, COOP_PREFERRED>, [U; N], #[stable(feature = "rust1", since = "1.0.0")], [(); core::alloc::co_alloc_metadata_num_slots_with_preference::<A>(COOP_PREFERRED)]: }
__impl_slice_eq1! { [A: Allocator, const COOP_PREFERRED: bool, const N: usize] Vec<T, A, COOP_PREFERRED>, &[U; N], #[stable(feature = "rust1", since = "1.0.0")], [(); core::alloc::co_alloc_metadata_num_slots_with_preference::<A>(COOP_PREFERRED)]: }

// NOTE: some less important impls are omitted to reduce code bloat
// FIXME(Centril): Reconsider this?
//__impl_slice_eq1! { [const N: usize] Vec<A>, &mut [B; N], }
//__impl_slice_eq1! { [const N: usize] [A; N], Vec<B>, }
//__impl_slice_eq1! { [const N: usize] &[A; N], Vec<B>, }
//__impl_slice_eq1! { [const N: usize] &mut [A; N], Vec<B>, }
//__impl_slice_eq1! { [const N: usize] Cow<'a, [A]>, [B; N], }
//__impl_slice_eq1! { [const N: usize] Cow<'a, [A]>, &[B; N], }
//__impl_slice_eq1! { [const N: usize] Cow<'a, [A]>, &mut [B; N], }
