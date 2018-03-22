//! Implements `Hash`.
#![allow(unused)]

macro_rules! impl_hash {
    ($id: ident, $elem_ty: ident) => {
        impl ::hash::Hash for $id {
            #[inline]
            fn hash<H: ::hash::Hasher>(&self, state: &mut H) {
                union A {
                    data: [$elem_ty; $id::lanes()],
                    vec: $id,
                }
                unsafe { A { vec: *self }.data.hash(state) }
            }
        }
    };
}

#[cfg(test)]
macro_rules! test_hash {
    ($id: ident, $elem_ty: ident) => {
        #[test]
        fn hash() {
            use coresimd::simd::$id;
            use std::collections::hash_map::DefaultHasher;
            use std::hash::{Hash, Hasher};
            use std::mem;
            type A = [$elem_ty; $id::lanes()];
            let a: A = [42 as $elem_ty; $id::lanes()];
            assert!(mem::size_of::<A>() == mem::size_of::<$id>());
            let mut a_hash = DefaultHasher::new();
            let mut v_hash = a_hash.clone();
            a.hash(&mut a_hash);

            let v = $id::splat(42 as $elem_ty);
            v.hash(&mut v_hash);
            assert_eq!(a_hash.finish(), v_hash.finish());
        }
    };
}
