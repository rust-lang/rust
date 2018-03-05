//! Implements portable bitwise vector reductions.

macro_rules! impl_bitwise_reductions {
    ($id:ident, $elem_ty:ident) => {
        impl $id {
            /// Lane-wise bitwise `and` of the vector elements.
            #[inline]
            pub fn and(self) -> $elem_ty {
                ReduceAnd::reduce_and(self)
            }
            /// Lane-wise bitwise `or` of the vector elements.
            #[inline]
            pub fn or(self) -> $elem_ty {
                ReduceOr::reduce_or(self)
            }
            /// Lane-wise bitwise `xor` of the vector elements.
            #[inline]
            pub fn xor(self) -> $elem_ty {
                ReduceXor::reduce_xor(self)
            }
        }
    }
}

macro_rules! impl_bool_bitwise_reductions {
    ($id:ident, $elem_ty:ident) => {
        impl $id {
            /// Lane-wise bitwise `and` of the vector elements.
            #[inline]
            pub fn and(self) -> $elem_ty {
                ReduceAnd::reduce_and(self) !=0
            }
            /// Lane-wise bitwise `or` of the vector elements.
            #[inline]
            pub fn or(self) -> $elem_ty {
                ReduceOr::reduce_or(self) != 0
            }
            /// Lane-wise bitwise `xor` of the vector elements.
            #[inline]
            pub fn xor(self) -> $elem_ty {
                ReduceXor::reduce_xor(self) != 0
            }
        }
    }
}


#[cfg(test)]
macro_rules! test_bitwise_reductions {
    ($id:ident, $true:expr) => {
        #[test]
        fn and() {
            let false_ = !$true;
            use ::coresimd::simd::$id;
            let v = $id::splat(false_);
            assert_eq!(v.and(), false_);
            let v = $id::splat($true);
            assert_eq!(v.and(), $true);
            let v = $id::splat(false_);
            let v = v.replace(0, $true);
            assert_eq!(v.and(), false_);
            let v = $id::splat($true);
            let v = v.replace(0, false_);
            assert_eq!(v.and(), false_);
        }
        #[test]
        fn or() {
            let false_ = !$true;
            use ::coresimd::simd::$id;
            let v = $id::splat(false_);
            assert_eq!(v.or(), false_);
            let v = $id::splat($true);
            assert_eq!(v.or(), $true);
            let v = $id::splat(false_);
            let v = v.replace(0, $true);
            assert_eq!(v.or(), $true);
            let v = $id::splat($true);
            let v = v.replace(0, false_);
            assert_eq!(v.or(), $true);
        }
        #[test]
        fn xor() {
            let false_ = !$true;
            use ::coresimd::simd::$id;
            let v = $id::splat(false_);
            assert_eq!(v.xor(), false_);
            let v = $id::splat($true);
            assert_eq!(v.xor(), false_);
            let v = $id::splat(false_);
            let v = v.replace(0, $true);
            assert_eq!(v.xor(), $true);
            let v = $id::splat($true);
            let v = v.replace(0, false_);
            assert_eq!(v.xor(), $true);
        }
    }
}
