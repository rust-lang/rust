//! Mask select method
#![allow(unused)]

/// Implements mask select method
macro_rules! impl_mask_select {
    ($id:ident, $elem_ty:ident, $elem_count:expr) => {
        impl $id {
            /// Selects elements of `a` and `b` using mask.
            ///
            /// For each lane, the result contains the element of `a` if the
            /// mask is true, and the element of `b` otherwise.
            #[inline]
            pub fn select<T>(self, a: T, b: T) -> T
            where
                T: super::api::Lanes<[u32; $elem_count]>,
            {
                use coresimd::simd_llvm::simd_select;
                unsafe { simd_select(self, a, b) }
            }
        }
    };
}

#[cfg(test)]
macro_rules! test_mask_select {
    ($mask_id:ident, $vec_id:ident, $elem_ty:ident) => {
        #[test]
        fn select() {
            use coresimd::simd::{$mask_id, $vec_id};
            let o = 1 as $elem_ty;
            let t = 2 as $elem_ty;

            let a = $vec_id::splat(o);
            let b = $vec_id::splat(t);
            let m = a.lt(b);
            assert_eq!(m.select(a, b), a);

            let m = b.lt(a);
            assert_eq!(m.select(b, a), a);

            let mut c = a;
            let mut d = b;
            let mut m_e = $mask_id::splat(false);
            for i in 0..$vec_id::lanes() {
                if i % 2 == 0 {
                    let c_tmp = c.extract(i);
                    c = c.replace(i, d.extract(i));
                    d = d.replace(i, c_tmp);
                } else {
                    m_e = m_e.replace(i, true);
                }
            }

            let m = c.lt(d);
            assert_eq!(m_e, m);
            assert_eq!(m.select(c, d), a);
        }
    };
}
