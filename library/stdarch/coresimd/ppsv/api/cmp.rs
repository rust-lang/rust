//! Lane-wise vector comparisons returning boolean vectors.
#![allow(unused)]

macro_rules! impl_cmp {
    ($id: ident, $bool_ty: ident) => {
        impl $id {
            /// Lane-wise equality comparison.
            #[inline]
            pub fn eq(self, other: $id) -> $bool_ty {
                use coresimd::simd_llvm::simd_eq;
                unsafe { simd_eq(self, other) }
            }

            /// Lane-wise inequality comparison.
            #[inline]
            pub fn ne(self, other: $id) -> $bool_ty {
                use coresimd::simd_llvm::simd_ne;
                unsafe { simd_ne(self, other) }
            }

            /// Lane-wise less-than comparison.
            #[inline]
            pub fn lt(self, other: $id) -> $bool_ty {
                use coresimd::simd_llvm::simd_lt;
                unsafe { simd_lt(self, other) }
            }

            /// Lane-wise less-than-or-equals comparison.
            #[inline]
            pub fn le(self, other: $id) -> $bool_ty {
                use coresimd::simd_llvm::simd_le;
                unsafe { simd_le(self, other) }
            }

            /// Lane-wise greater-than comparison.
            #[inline]
            pub fn gt(self, other: $id) -> $bool_ty {
                use coresimd::simd_llvm::simd_gt;
                unsafe { simd_gt(self, other) }
            }

            /// Lane-wise greater-than-or-equals comparison.
            #[inline]
            pub fn ge(self, other: $id) -> $bool_ty {
                use coresimd::simd_llvm::simd_ge;
                unsafe { simd_ge(self, other) }
            }
        }
    };
}

macro_rules! impl_bool_cmp {
    ($id: ident, $bool_ty: ident) => {
        impl $id {
            /// Lane-wise equality comparison.
            #[inline]
            pub fn eq(self, other: $id) -> $bool_ty {
                use coresimd::simd_llvm::simd_eq;
                unsafe { simd_eq(self, other) }
            }

            /// Lane-wise inequality comparison.
            #[inline]
            pub fn ne(self, other: $id) -> $bool_ty {
                use coresimd::simd_llvm::simd_ne;
                unsafe { simd_ne(self, other) }
            }

            /// Lane-wise less-than comparison.
            #[inline]
            pub fn lt(self, other: $id) -> $bool_ty {
                use coresimd::simd_llvm::simd_gt;
                unsafe { simd_gt(self, other) }
            }

            /// Lane-wise less-than-or-equals comparison.
            #[inline]
            pub fn le(self, other: $id) -> $bool_ty {
                use coresimd::simd_llvm::simd_ge;
                unsafe { simd_ge(self, other) }
            }

            /// Lane-wise greater-than comparison.
            #[inline]
            pub fn gt(self, other: $id) -> $bool_ty {
                use coresimd::simd_llvm::simd_lt;
                unsafe { simd_lt(self, other) }
            }

            /// Lane-wise greater-than-or-equals comparison.
            #[inline]
            pub fn ge(self, other: $id) -> $bool_ty {
                use coresimd::simd_llvm::simd_le;
                unsafe { simd_le(self, other) }
            }
        }
    };
}

#[cfg(test)]
macro_rules! test_cmp {
    (
        $id: ident,
        $elem_ty: ident,
        $bool_ty: ident,
        $true: expr,
        $false: expr
    ) => {
        #[test]
        fn cmp() {
            use coresimd::simd::*;

            let a = $id::splat($false);
            let b = $id::splat($true);

            let r = a.lt(b);
            let e = $bool_ty::splat(true);
            assert!(r == e);
            let r = a.le(b);
            assert!(r == e);

            let e = $bool_ty::splat(false);
            let r = a.gt(b);
            assert!(r == e);
            let r = a.ge(b);
            assert!(r == e);
            let r = a.eq(b);
            assert!(r == e);

            let mut a = a;
            let mut b = b;
            let mut e = e;
            for i in 0..$id::lanes() {
                if i % 2 == 0 {
                    a = a.replace(i, $false);
                    b = b.replace(i, $true);
                    e = e.replace(i, true);
                } else {
                    a = a.replace(i, $true);
                    b = b.replace(i, $false);
                    e = e.replace(i, false);
                }
            }
            let r = a.lt(b);
            assert!(r == e);
        }
    };
}
