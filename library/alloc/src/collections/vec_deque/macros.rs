macro_rules! __impl_slice_eq1 {
    ([$($vars:tt)*] $lhs:ty, $rhs:ty, $($constraints:tt)*) => {
        #[stable(feature = "vec_deque_partial_eq_slice", since = "1.17.0")]
        impl<A, B, $($vars)*> PartialEq<$rhs> for $lhs
        where
            A: PartialEq<B>,
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
