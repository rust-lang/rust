macro_rules! __impl_slice_eq1 {
    ([$($vars:tt)*] $lhs:ty, $rhs:ty, $($constraints:tt)*) => {
        #[stable(feature = "vec_deque_partial_eq_slice", since = "1.17.0")]
        impl<T, U, A: Allocator, const COOP_PREFERRED: bool, $($vars)*> PartialEq<$rhs> for $lhs
        where
            T: PartialEq<U>,
            [(); core::alloc::co_alloc_metadata_num_slots_with_preference::<A>(COOP_PREFERRED)]:,
            $($constraints)*
        {
            fn eq(&self, other: &$rhs) -> bool {
                if self.len() != other.len() {
                    return false;
                }
                let (sa, sb) = self.as_slices();
                let (oa, ob) = other[..].split_at(sa.len());
                sa == oa && sb == ob
            }
        }
    }
}
