//use core::alloc;
use crate::alloc::Allocator;
#[cfg(not(no_global_oom_handling))]
use crate::borrow::Cow;
use crate::co_alloc::CoAllocPref;

use super::Vec;

macro_rules! __impl_slice_eq1 {
    ([$($vars:tt)*] $lhs:ty, $rhs:ty, #[$stability:meta], $($constraints:tt)*) => {
        #[$stability]
        #[allow(unused_braces)]
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

__impl_slice_eq1! { [A1: Allocator, A2: Allocator, const CO_ALLOC_PREF1: crate::co_alloc::CoAllocPref, const CO_ALLOC_PREF2: crate::co_alloc::CoAllocPref] Vec<T, A1, CO_ALLOC_PREF1>, Vec<U, A2, CO_ALLOC_PREF2>, #[stable(feature = "rust1", since = "1.0.0")], [(); {crate::meta_num_slots!(A1, CO_ALLOC_PREF1)}]:, [(); {crate::meta_num_slots!(A2, CO_ALLOC_PREF2)}]: }
__impl_slice_eq1! { [A: Allocator, const CO_ALLOC_PREF: CoAllocPref] Vec<T, A, CO_ALLOC_PREF>, &[U], #[stable(feature = "rust1", since = "1.0.0")], [(); {crate::meta_num_slots!(A, CO_ALLOC_PREF)}]: }
__impl_slice_eq1! { [A: Allocator, const CO_ALLOC_PREF: CoAllocPref] Vec<T, A, CO_ALLOC_PREF>, &mut [U], #[stable(feature = "rust1", since = "1.0.0")], [(); {crate::meta_num_slots!(A, CO_ALLOC_PREF)}]: }
__impl_slice_eq1! { [A: Allocator, const CO_ALLOC_PREF: CoAllocPref] &[T], Vec<U, A, CO_ALLOC_PREF>, #[stable(feature = "partialeq_vec_for_ref_slice", since = "1.46.0")], [(); {crate::meta_num_slots!(A, CO_ALLOC_PREF)}]: }
__impl_slice_eq1! { [A: Allocator, const CO_ALLOC_PREF: CoAllocPref] &mut [T], Vec<U, A, CO_ALLOC_PREF>, #[stable(feature = "partialeq_vec_for_ref_slice", since = "1.46.0")], [(); {crate::meta_num_slots!(A, CO_ALLOC_PREF)}]: }
__impl_slice_eq1! { [A: Allocator, const CO_ALLOC_PREF: CoAllocPref] Vec<T, A, CO_ALLOC_PREF>, [U], #[stable(feature = "partialeq_vec_for_slice", since = "1.48.0")], [(); {crate::meta_num_slots!(A, CO_ALLOC_PREF)}]:  }
__impl_slice_eq1! { [A: Allocator, const CO_ALLOC_PREF: CoAllocPref] [T], Vec<U, A, CO_ALLOC_PREF>, #[stable(feature = "partialeq_vec_for_slice", since = "1.48.0")], [(); {crate::meta_num_slots!(A, CO_ALLOC_PREF)}]:  }
#[cfg(not(no_global_oom_handling))]
__impl_slice_eq1! { [A: Allocator, const CO_ALLOC_PREF: CoAllocPref] Cow<'_, [T]>, Vec<U, A, CO_ALLOC_PREF>, #[stable(feature = "rust1", since = "1.0.0")], T: Clone, [(); {crate::meta_num_slots!(A, CO_ALLOC_PREF)}]: }
#[cfg(not(no_global_oom_handling))]
__impl_slice_eq1! { [] Cow<'_, [T]>, &[U], #[stable(feature = "rust1", since = "1.0.0")], T: Clone }
#[cfg(not(no_global_oom_handling))]
__impl_slice_eq1! { [] Cow<'_, [T]>, &mut [U], #[stable(feature = "rust1", since = "1.0.0")], T: Clone }
__impl_slice_eq1! { [A: Allocator, const CO_ALLOC_PREF: CoAllocPref, const N: usize] Vec<T, A, CO_ALLOC_PREF>, [U; N], #[stable(feature = "rust1", since = "1.0.0")], [(); {crate::meta_num_slots!(A, CO_ALLOC_PREF)}]: }
__impl_slice_eq1! { [A: Allocator, const CO_ALLOC_PREF: CoAllocPref, const N: usize] Vec<T, A, CO_ALLOC_PREF>, &[U; N], #[stable(feature = "rust1", since = "1.0.0")], [(); {crate::meta_num_slots!(A, CO_ALLOC_PREF)}]: }

// NOTE: some less important impls are omitted to reduce code bloat
// FIXME(Centril): Reconsider this?
//__impl_slice_eq1! { [const N: usize] Vec<A>, &mut [B; N], }
//__impl_slice_eq1! { [const N: usize] [A; N], Vec<B>, }
//__impl_slice_eq1! { [const N: usize] &[A; N], Vec<B>, }
//__impl_slice_eq1! { [const N: usize] &mut [A; N], Vec<B>, }
//__impl_slice_eq1! { [const N: usize] Cow<'a, [A]>, [B; N], }
//__impl_slice_eq1! { [const N: usize] Cow<'a, [A]>, &[B; N], }
//__impl_slice_eq1! { [const N: usize] Cow<'a, [A]>, &mut [B; N], }
