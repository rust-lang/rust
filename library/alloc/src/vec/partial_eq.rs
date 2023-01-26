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

__impl_slice_eq1! { [A1: Allocator, A2: Allocator, const COOP_PREF1: bool, const COOP_PREF2: bool] Vec<T, A1, COOP_PREF1>, Vec<U, A2, COOP_PREF2>, #[stable(feature = "rust1", since = "1.0.0")], [(); core::alloc::co_alloc_metadata_num_slots_with_preference::<A1>(COOP_PREF1)]:, [(); core::alloc::co_alloc_metadata_num_slots_with_preference::<A2>(COOP_PREF2)]: }
__impl_slice_eq1! { [A: Allocator, const COOP_PREF: bool] Vec<T, A, COOP_PREF>, &[U], #[stable(feature = "rust1", since = "1.0.0")], [(); core::alloc::co_alloc_metadata_num_slots_with_preference::<A>(COOP_PREF)]: }
__impl_slice_eq1! { [A: Allocator, const COOP_PREF: bool] Vec<T, A, COOP_PREF>, &mut [U], #[stable(feature = "rust1", since = "1.0.0")], [(); core::alloc::co_alloc_metadata_num_slots_with_preference::<A>(COOP_PREF)]: }
__impl_slice_eq1! { [A: Allocator, const COOP_PREF: bool] &[T], Vec<U, A, COOP_PREF>, #[stable(feature = "partialeq_vec_for_ref_slice", since = "1.46.0")], [(); core::alloc::co_alloc_metadata_num_slots_with_preference::<A>(COOP_PREF)]: }
__impl_slice_eq1! { [A: Allocator, const COOP_PREF: bool] &mut [T], Vec<U, A, COOP_PREF>, #[stable(feature = "partialeq_vec_for_ref_slice", since = "1.46.0")], [(); core::alloc::co_alloc_metadata_num_slots_with_preference::<A>(COOP_PREF)]: }
__impl_slice_eq1! { [A: Allocator, const COOP_PREF: bool] Vec<T, A, COOP_PREF>, [U], #[stable(feature = "partialeq_vec_for_slice", since = "1.48.0")], [(); core::alloc::co_alloc_metadata_num_slots_with_preference::<A>(COOP_PREF)]:  }
__impl_slice_eq1! { [A: Allocator, const COOP_PREF: bool] [T], Vec<U, A, COOP_PREF>, #[stable(feature = "partialeq_vec_for_slice", since = "1.48.0")], [(); core::alloc::co_alloc_metadata_num_slots_with_preference::<A>(COOP_PREF)]:  }
#[cfg(not(no_global_oom_handling))]
__impl_slice_eq1! { [A: Allocator, const COOP_PREF: bool] Cow<'_, [T]>, Vec<U, A, COOP_PREF>, #[stable(feature = "rust1", since = "1.0.0")], T: Clone, [(); core::alloc::co_alloc_metadata_num_slots_with_preference::<A>(COOP_PREF)]: }
#[cfg(not(no_global_oom_handling))]
__impl_slice_eq1! { [] Cow<'_, [T]>, &[U], #[stable(feature = "rust1", since = "1.0.0")], T: Clone }
#[cfg(not(no_global_oom_handling))]
__impl_slice_eq1! { [] Cow<'_, [T]>, &mut [U], #[stable(feature = "rust1", since = "1.0.0")], T: Clone }
__impl_slice_eq1! { [A: Allocator, const COOP_PREF: bool, const N: usize] Vec<T, A, COOP_PREF>, [U; N], #[stable(feature = "rust1", since = "1.0.0")], [(); core::alloc::co_alloc_metadata_num_slots_with_preference::<A>(COOP_PREF)]: }
__impl_slice_eq1! { [A: Allocator, const COOP_PREF: bool, const N: usize] Vec<T, A, COOP_PREF>, &[U; N], #[stable(feature = "rust1", since = "1.0.0")], [(); core::alloc::co_alloc_metadata_num_slots_with_preference::<A>(COOP_PREF)]: }

// NOTE: some less important impls are omitted to reduce code bloat
// FIXME(Centril): Reconsider this?
//__impl_slice_eq1! { [const N: usize] Vec<A>, &mut [B; N], }
//__impl_slice_eq1! { [const N: usize] [A; N], Vec<B>, }
//__impl_slice_eq1! { [const N: usize] &[A; N], Vec<B>, }
//__impl_slice_eq1! { [const N: usize] &mut [A; N], Vec<B>, }
//__impl_slice_eq1! { [const N: usize] Cow<'a, [A]>, [B; N], }
//__impl_slice_eq1! { [const N: usize] Cow<'a, [A]>, &[B; N], }
//__impl_slice_eq1! { [const N: usize] Cow<'a, [A]>, &mut [B; N], }
