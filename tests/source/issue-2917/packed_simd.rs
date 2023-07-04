// rustfmt-wrap_comments: true
//! Implements `From` and `Into` for vector types.

macro_rules! impl_from_vector {
    ([$elem_ty:ident; $elem_count:expr]: $id:ident | $test_tt:tt | $source:ident) => {
        impl From<$source> for $id {
            #[inline]
            fn from(source: $source) -> Self {
                fn static_assert_same_number_of_lanes<T, U>()
                where
                    T: crate::sealed::Simd,
                    U: crate::sealed::Simd<LanesType = T::LanesType>,
                {
                }
                use llvm::simd_cast;
                static_assert_same_number_of_lanes::<$id, $source>();
                Simd(unsafe { simd_cast(source.0) })
            }
        }

        // FIXME: `Into::into` is not inline, but due to
                // the blanket impl in `std`, which is not
                // marked `default`, we cannot override it here with
                // specialization.
                /*
                impl Into<$id> for $source {
                    #[inline]
                    fn into(self) -> $id {
                        unsafe { simd_cast(self) }
                    }
                }
                */

        test_if!{
            $test_tt:
            interpolate_idents! {
                mod [$id _from_ $source] {
                    use super::*;
                    #[test]
                    fn from() {
                        assert_eq!($id::lanes(), $source::lanes());
                        let source: $source = Default::default();
                        let vec: $id = Default::default();

                        let e = $id::from(source);
                        assert_eq!(e, vec);

                        let e: $id = source.into();
                        assert_eq!(e, vec);
                    }
                }
            }
        }
    };
}

macro_rules! impl_from_vectors {
    ([$elem_ty:ident; $elem_count:expr]: $id:ident | $test_tt:tt | $($source:ident),*) => {
        $(
            impl_from_vector!([$elem_ty; $elem_count]: $id | $test_tt | $source);
        )*
    }
}
