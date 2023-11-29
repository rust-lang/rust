use crate::alloc::{Allocator, failure_handling::FailureHandling};
#[cfg(not(no_global_oom_handling))]
use crate::borrow::Cow;

use super::Vec;

macro_rules! __impl_slice_eq1 {
    ([$($vars:tt)*] $lhs:ty, $rhs:ty $(where $ty:ty: $bound:ident)?, #[$stability:meta]) => {
        #[$stability]
        impl<T, U, $($vars)*> PartialEq<$rhs> for $lhs
        where
            T: PartialEq<U>,
            $($ty: $bound)?
        {
            #[inline]
            fn eq(&self, other: &$rhs) -> bool { self[..] == other[..] }
            #[inline]
            fn ne(&self, other: &$rhs) -> bool { self[..] != other[..] }
        }
    }
}

__impl_slice_eq1! { [A1: Allocator, A2: Allocator, FH1: FailureHandling, FH2: FailureHandling] Vec<T, A1, FH1>, Vec<U, A2, FH2>, #[stable(feature = "rust1", since = "1.0.0")] }
__impl_slice_eq1! { [A: Allocator, FH: FailureHandling] Vec<T, A, FH>, &[U], #[stable(feature = "rust1", since = "1.0.0")] }
__impl_slice_eq1! { [A: Allocator, FH: FailureHandling] Vec<T, A, FH>, &mut [U], #[stable(feature = "rust1", since = "1.0.0")] }
__impl_slice_eq1! { [A: Allocator, FH: FailureHandling] &[T], Vec<U, A, FH>, #[stable(feature = "partialeq_vec_for_ref_slice", since = "1.46.0")] }
__impl_slice_eq1! { [A: Allocator, FH: FailureHandling] &mut [T], Vec<U, A, FH>, #[stable(feature = "partialeq_vec_for_ref_slice", since = "1.46.0")] }
__impl_slice_eq1! { [A: Allocator, FH: FailureHandling] Vec<T, A, FH>, [U], #[stable(feature = "partialeq_vec_for_slice", since = "1.48.0")]  }
__impl_slice_eq1! { [A: Allocator, FH: FailureHandling] [T], Vec<U, A, FH>, #[stable(feature = "partialeq_vec_for_slice", since = "1.48.0")]  }
#[cfg(not(no_global_oom_handling))]
__impl_slice_eq1! { [A: Allocator, FH: FailureHandling] Cow<'_, [T]>, Vec<U, A, FH> where T: Clone, #[stable(feature = "rust1", since = "1.0.0")] }
#[cfg(not(no_global_oom_handling))]
__impl_slice_eq1! { [] Cow<'_, [T]>, &[U] where T: Clone, #[stable(feature = "rust1", since = "1.0.0")] }
#[cfg(not(no_global_oom_handling))]
__impl_slice_eq1! { [] Cow<'_, [T]>, &mut [U] where T: Clone, #[stable(feature = "rust1", since = "1.0.0")] }
__impl_slice_eq1! { [A: Allocator, FH: FailureHandling, const N: usize] Vec<T, A, FH>, [U; N], #[stable(feature = "rust1", since = "1.0.0")] }
__impl_slice_eq1! { [A: Allocator, FH: FailureHandling, const N: usize] Vec<T, A, FH>, &[U; N], #[stable(feature = "rust1", since = "1.0.0")] }

// NOTE: some less important impls are omitted to reduce code bloat
// FIXME(Centril): Reconsider this?
//__impl_slice_eq1! { [const N: usize] Vec<A>, &mut [B; N], }
//__impl_slice_eq1! { [const N: usize] [A; N], Vec<B>, }
//__impl_slice_eq1! { [const N: usize] &[A; N], Vec<B>, }
//__impl_slice_eq1! { [const N: usize] &mut [A; N], Vec<B>, }
//__impl_slice_eq1! { [const N: usize] Cow<'a, [A]>, [B; N], }
//__impl_slice_eq1! { [const N: usize] Cow<'a, [A]>, &[B; N], }
//__impl_slice_eq1! { [const N: usize] Cow<'a, [A]>, &mut [B; N], }
