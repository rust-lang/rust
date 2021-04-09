use crate::LanesAtMost32;

/// Checks if the right-hand side argument of a left- or right-shift would cause overflow.
fn invalid_shift_rhs<T>(rhs: T) -> bool
where
    T: Default + PartialOrd + core::convert::TryFrom<usize>,
    <T as core::convert::TryFrom<usize>>::Error: core::fmt::Debug,
{
    let bits_in_type = T::try_from(8 * core::mem::size_of::<T>()).unwrap();
    rhs < T::default() || rhs >= bits_in_type
}

/// Automatically implements operators over references in addition to the provided operator.
macro_rules! impl_ref_ops {
    // binary op
    {
        impl<const $lanes:ident: usize> core::ops::$trait:ident<$rhs:ty> for $type:ty
        where
            $($bound:path: LanesAtMost32,)*
        {
            type Output = $output:ty;

            $(#[$attrs:meta])*
            fn $fn:ident($self_tok:ident, $rhs_arg:ident: $rhs_arg_ty:ty) -> Self::Output $body:tt
        }
    } => {
        impl<const $lanes: usize> core::ops::$trait<$rhs> for $type
        where
            $($bound: LanesAtMost32,)*
        {
            type Output = $output;

            $(#[$attrs])*
            fn $fn($self_tok, $rhs_arg: $rhs_arg_ty) -> Self::Output $body
        }

        impl<const $lanes: usize> core::ops::$trait<&'_ $rhs> for $type
        where
            $($bound: LanesAtMost32,)*
        {
            type Output = <$type as core::ops::$trait<$rhs>>::Output;

            $(#[$attrs])*
            fn $fn($self_tok, $rhs_arg: &$rhs) -> Self::Output {
                core::ops::$trait::$fn($self_tok, *$rhs_arg)
            }
        }

        impl<const $lanes: usize> core::ops::$trait<$rhs> for &'_ $type
        where
            $($bound: LanesAtMost32,)*
        {
            type Output = <$type as core::ops::$trait<$rhs>>::Output;

            $(#[$attrs])*
            fn $fn($self_tok, $rhs_arg: $rhs) -> Self::Output {
                core::ops::$trait::$fn(*$self_tok, $rhs_arg)
            }
        }

        impl<const $lanes: usize> core::ops::$trait<&'_ $rhs> for &'_ $type
        where
            $($bound: LanesAtMost32,)*
        {
            type Output = <$type as core::ops::$trait<$rhs>>::Output;

            $(#[$attrs])*
            fn $fn($self_tok, $rhs_arg: &$rhs) -> Self::Output {
                core::ops::$trait::$fn(*$self_tok, *$rhs_arg)
            }
        }
    };

    // binary assignment op
    {
        impl<const $lanes:ident: usize> core::ops::$trait:ident<$rhs:ty> for $type:ty
        where
            $($bound:path: LanesAtMost32,)*
        {
            $(#[$attrs:meta])*
            fn $fn:ident(&mut $self_tok:ident, $rhs_arg:ident: $rhs_arg_ty:ty) $body:tt
        }
    } => {
        impl<const $lanes: usize> core::ops::$trait<$rhs> for $type
        where
            $($bound: LanesAtMost32,)*
        {
            $(#[$attrs])*
            fn $fn(&mut $self_tok, $rhs_arg: $rhs_arg_ty) $body
        }

        impl<const $lanes: usize> core::ops::$trait<&'_ $rhs> for $type
        where
            $($bound: LanesAtMost32,)*
        {
            $(#[$attrs])*
            fn $fn(&mut $self_tok, $rhs_arg: &$rhs_arg_ty) {
                core::ops::$trait::$fn($self_tok, *$rhs_arg)
            }
        }
    };

    // unary op
    {
        impl<const $lanes:ident: usize> core::ops::$trait:ident for $type:ty
        where
            $($bound:path: LanesAtMost32,)*
        {
            type Output = $output:ty;
            fn $fn:ident($self_tok:ident) -> Self::Output $body:tt
        }
    } => {
        impl<const $lanes: usize> core::ops::$trait for $type
        where
            $($bound: LanesAtMost32,)*
        {
            type Output = $output;
            fn $fn($self_tok) -> Self::Output $body
        }

        impl<const $lanes: usize> core::ops::$trait for &'_ $type
        where
            $($bound: LanesAtMost32,)*
        {
            type Output = <$type as core::ops::$trait>::Output;
            fn $fn($self_tok) -> Self::Output {
                core::ops::$trait::$fn(*$self_tok)
            }
        }
    }
}

/// Automatically implements operators over vectors and scalars for a particular vector.
macro_rules! impl_op {
    { impl Add for $type:ident, $scalar:ty } => {
        impl_op! { @binary $type, $scalar, Add::add, AddAssign::add_assign, simd_add }
    };
    { impl Sub for $type:ident, $scalar:ty } => {
        impl_op! { @binary $type, $scalar, Sub::sub, SubAssign::sub_assign, simd_sub }
    };
    { impl Mul for $type:ident, $scalar:ty } => {
        impl_op! { @binary $type, $scalar, Mul::mul, MulAssign::mul_assign, simd_mul }
    };
    { impl Div for $type:ident, $scalar:ty } => {
        impl_op! { @binary $type, $scalar, Div::div, DivAssign::div_assign, simd_div }
    };
    { impl Rem for $type:ident, $scalar:ty } => {
        impl_op! { @binary $type, $scalar, Rem::rem, RemAssign::rem_assign, simd_rem }
    };
    { impl Shl for $type:ident, $scalar:ty } => {
        impl_op! { @binary $type, $scalar, Shl::shl, ShlAssign::shl_assign, simd_shl }
    };
    { impl Shr for $type:ident, $scalar:ty } => {
        impl_op! { @binary $type, $scalar, Shr::shr, ShrAssign::shr_assign, simd_shr }
    };
    { impl BitAnd for $type:ident, $scalar:ty } => {
        impl_op! { @binary $type, $scalar, BitAnd::bitand, BitAndAssign::bitand_assign, simd_and }
    };
    { impl BitOr for $type:ident, $scalar:ty } => {
        impl_op! { @binary $type, $scalar, BitOr::bitor, BitOrAssign::bitor_assign, simd_or }
    };
    { impl BitXor for $type:ident, $scalar:ty } => {
        impl_op! { @binary $type, $scalar, BitXor::bitxor, BitXorAssign::bitxor_assign, simd_xor }
    };

    { impl Not for $type:ident, $scalar:ty } => {
        impl_ref_ops! {
            impl<const LANES: usize> core::ops::Not for crate::$type<LANES>
            where
                crate::$type<LANES>: LanesAtMost32,
            {
                type Output = Self;
                fn not(self) -> Self::Output {
                    self ^ Self::splat(!<$scalar>::default())
                }
            }
        }
    };

    { impl Neg for $type:ident, $scalar:ty } => {
        impl_ref_ops! {
            impl<const LANES: usize> core::ops::Neg for crate::$type<LANES>
            where
                crate::$type<LANES>: LanesAtMost32,
            {
                type Output = Self;
                fn neg(self) -> Self::Output {
                    Self::splat(0) - self
                }
            }
        }
    };

    { impl Neg for $type:ident, $scalar:ty, @float } => {
        impl_ref_ops! {
            impl<const LANES: usize> core::ops::Neg for crate::$type<LANES>
            where
                crate::$type<LANES>: LanesAtMost32,
                crate::SimdU32<LANES>: LanesAtMost32,
                crate::SimdU64<LANES>: LanesAtMost32,
            {
                type Output = Self;
                fn neg(self) -> Self::Output {
                    // FIXME: Replace this with fneg intrinsic once available.
                    // https://github.com/rust-lang/stdsimd/issues/32
                    Self::from_bits(Self::splat(-0.0).to_bits() ^ self.to_bits())
                }
            }
        }
    };

    { impl Index for $type:ident, $scalar:ty } => {
        impl<I, const LANES: usize> core::ops::Index<I> for crate::$type<LANES>
        where
            Self: LanesAtMost32,
            I: core::slice::SliceIndex<[$scalar]>,
        {
            type Output = I::Output;
            fn index(&self, index: I) -> &Self::Output {
                let slice: &[_] = self.as_ref();
                &slice[index]
            }
        }

        impl<I, const LANES: usize> core::ops::IndexMut<I> for crate::$type<LANES>
        where
            Self: LanesAtMost32,
            I: core::slice::SliceIndex<[$scalar]>,
        {
            fn index_mut(&mut self, index: I) -> &mut Self::Output {
                let slice: &mut [_] = self.as_mut();
                &mut slice[index]
            }
        }
    };

    // generic binary op with assignment when output is `Self`
    { @binary $type:ident, $scalar:ty, $trait:ident :: $trait_fn:ident, $assign_trait:ident :: $assign_trait_fn:ident, $intrinsic:ident } => {
        impl_ref_ops! {
            impl<const LANES: usize> core::ops::$trait<Self> for crate::$type<LANES>
            where
                crate::$type<LANES>: LanesAtMost32,
            {
                type Output = Self;

                #[inline]
                fn $trait_fn(self, rhs: Self) -> Self::Output {
                    unsafe {
                        crate::intrinsics::$intrinsic(self, rhs)
                    }
                }
            }
        }

        impl_ref_ops! {
            impl<const LANES: usize> core::ops::$trait<$scalar> for crate::$type<LANES>
            where
                crate::$type<LANES>: LanesAtMost32,
            {
                type Output = Self;

                #[inline]
                fn $trait_fn(self, rhs: $scalar) -> Self::Output {
                    core::ops::$trait::$trait_fn(self, Self::splat(rhs))
                }
            }
        }

        impl_ref_ops! {
            impl<const LANES: usize> core::ops::$trait<crate::$type<LANES>> for $scalar
            where
                crate::$type<LANES>: LanesAtMost32,
            {
                type Output = crate::$type<LANES>;

                #[inline]
                fn $trait_fn(self, rhs: crate::$type<LANES>) -> Self::Output {
                    core::ops::$trait::$trait_fn(crate::$type::splat(self), rhs)
                }
            }
        }

        impl_ref_ops! {
            impl<const LANES: usize> core::ops::$assign_trait<Self> for crate::$type<LANES>
            where
                crate::$type<LANES>: LanesAtMost32,
            {
                #[inline]
                fn $assign_trait_fn(&mut self, rhs: Self) {
                    unsafe {
                        *self = crate::intrinsics::$intrinsic(*self, rhs);
                    }
                }
            }
        }

        impl_ref_ops! {
            impl<const LANES: usize> core::ops::$assign_trait<$scalar> for crate::$type<LANES>
            where
                crate::$type<LANES>: LanesAtMost32,
            {
                #[inline]
                fn $assign_trait_fn(&mut self, rhs: $scalar) {
                    core::ops::$assign_trait::$assign_trait_fn(self, Self::splat(rhs));
                }
            }
        }
    };
}

/// Implements floating-point operators for the provided types.
macro_rules! impl_float_ops {
    { $($scalar:ty => $($vector:ident),*;)* } => {
        $( // scalar
            $( // vector
                impl_op! { impl Add for $vector, $scalar }
                impl_op! { impl Sub for $vector, $scalar }
                impl_op! { impl Mul for $vector, $scalar }
                impl_op! { impl Div for $vector, $scalar }
                impl_op! { impl Rem for $vector, $scalar }
                impl_op! { impl Neg for $vector, $scalar, @float }
                impl_op! { impl Index for $vector, $scalar }
            )*
        )*
    };
}

/// Implements unsigned integer operators for the provided types.
macro_rules! impl_unsigned_int_ops {
    { $($scalar:ty => $($vector:ident),*;)* } => {
        $( // scalar
            $( // vector
                impl_op! { impl Add for $vector, $scalar }
                impl_op! { impl Sub for $vector, $scalar }
                impl_op! { impl Mul for $vector, $scalar }
                impl_op! { impl BitAnd for $vector, $scalar }
                impl_op! { impl BitOr  for $vector, $scalar }
                impl_op! { impl BitXor for $vector, $scalar }
                impl_op! { impl Not for $vector, $scalar }
                impl_op! { impl Index for $vector, $scalar }

                // Integers panic on divide by 0
                impl_ref_ops! {
                    impl<const LANES: usize> core::ops::Div<Self> for crate::$vector<LANES>
                    where
                        crate::$vector<LANES>: LanesAtMost32,
                    {
                        type Output = Self;

                        #[inline]
                        fn div(self, rhs: Self) -> Self::Output {
                            if rhs.as_slice()
                                .iter()
                                .any(|x| *x == 0)
                            {
                                panic!("attempt to divide by zero");
                            }

                            // Guards for div(MIN, -1),
                            // this check only applies to signed ints
                            if <$scalar>::MIN != 0 && self.as_slice().iter()
                                    .zip(rhs.as_slice().iter())
                                    .any(|(x,y)| *x == <$scalar>::MIN && *y == -1 as _) {
                                panic!("attempt to divide with overflow");
                            }
                            unsafe { crate::intrinsics::simd_div(self, rhs) }
                        }
                    }
                }

                impl_ref_ops! {
                    impl<const LANES: usize> core::ops::Div<$scalar> for crate::$vector<LANES>
                    where
                        crate::$vector<LANES>: LanesAtMost32,
                    {
                        type Output = Self;

                        #[inline]
                        fn div(self, rhs: $scalar) -> Self::Output {
                            if rhs == 0 {
                                panic!("attempt to divide by zero");
                            }
                            if <$scalar>::MIN != 0 &&
                                self.as_slice().iter().any(|x| *x == <$scalar>::MIN) &&
                                rhs == -1 as _ {
                                    panic!("attempt to divide with overflow");
                            }
                            let rhs = Self::splat(rhs);
                            unsafe { crate::intrinsics::simd_div(self, rhs) }
                        }
                    }
                }

                impl_ref_ops! {
                    impl<const LANES: usize> core::ops::Div<crate::$vector<LANES>> for $scalar
                    where
                        crate::$vector<LANES>: LanesAtMost32,
                    {
                        type Output = crate::$vector<LANES>;

                        #[inline]
                        fn div(self, rhs: crate::$vector<LANES>) -> Self::Output {
                            crate::$vector::splat(self) / rhs
                        }
                    }
                }

                impl_ref_ops! {
                    impl<const LANES: usize> core::ops::DivAssign<Self> for crate::$vector<LANES>
                    where
                        crate::$vector<LANES>: LanesAtMost32,
                    {
                        #[inline]
                        fn div_assign(&mut self, rhs: Self) {
                            *self = *self / rhs;
                        }
                    }
                }

                impl_ref_ops! {
                    impl<const LANES: usize> core::ops::DivAssign<$scalar> for crate::$vector<LANES>
                    where
                        crate::$vector<LANES>: LanesAtMost32,
                    {
                        #[inline]
                        fn div_assign(&mut self, rhs: $scalar) {
                            *self = *self / rhs;
                        }
                    }
                }

                // remainder panics on zero divisor
                impl_ref_ops! {
                    impl<const LANES: usize> core::ops::Rem<Self> for crate::$vector<LANES>
                    where
                        crate::$vector<LANES>: LanesAtMost32,
                    {
                        type Output = Self;

                        #[inline]
                        fn rem(self, rhs: Self) -> Self::Output {
                            if rhs.as_slice()
                                .iter()
                                .any(|x| *x == 0)
                            {
                                panic!("attempt to calculate the remainder with a divisor of zero");
                            }

                            // Guards for rem(MIN, -1)
                            // this branch applies the check only to signed ints
                            if <$scalar>::MIN != 0 && self.as_slice().iter()
                                    .zip(rhs.as_slice().iter())
                                    .any(|(x,y)| *x == <$scalar>::MIN && *y == -1 as _) {
                                panic!("attempt to calculate the remainder with overflow");
                            }
                            unsafe { crate::intrinsics::simd_rem(self, rhs) }
                        }
                    }
                }

                impl_ref_ops! {
                    impl<const LANES: usize> core::ops::Rem<$scalar> for crate::$vector<LANES>
                    where
                        crate::$vector<LANES>: LanesAtMost32,
                    {
                        type Output = Self;

                        #[inline]
                        fn rem(self, rhs: $scalar) -> Self::Output {
                            if rhs == 0 {
                                panic!("attempt to calculate the remainder with a divisor of zero");
                            }
                            if <$scalar>::MIN != 0 &&
                                self.as_slice().iter().any(|x| *x == <$scalar>::MIN) &&
                                rhs == -1 as _ {
                                    panic!("attempt to calculate the remainder with overflow");
                            }
                            let rhs = Self::splat(rhs);
                            unsafe { crate::intrinsics::simd_rem(self, rhs) }
                        }
                    }
                }

                impl_ref_ops! {
                    impl<const LANES: usize> core::ops::Rem<crate::$vector<LANES>> for $scalar
                    where
                        crate::$vector<LANES>: LanesAtMost32,
                    {
                        type Output = crate::$vector<LANES>;

                        #[inline]
                        fn rem(self, rhs: crate::$vector<LANES>) -> Self::Output {
                            crate::$vector::splat(self) % rhs
                        }
                    }
                }

                impl_ref_ops! {
                    impl<const LANES: usize> core::ops::RemAssign<Self> for crate::$vector<LANES>
                    where
                        crate::$vector<LANES>: LanesAtMost32,
                    {
                        #[inline]
                        fn rem_assign(&mut self, rhs: Self) {
                            *self = *self % rhs;
                        }
                    }
                }

                impl_ref_ops! {
                    impl<const LANES: usize> core::ops::RemAssign<$scalar> for crate::$vector<LANES>
                    where
                        crate::$vector<LANES>: LanesAtMost32,
                    {
                        #[inline]
                        fn rem_assign(&mut self, rhs: $scalar) {
                            *self = *self % rhs;
                        }
                    }
                }

                // shifts panic on overflow
                impl_ref_ops! {
                    impl<const LANES: usize> core::ops::Shl<Self> for crate::$vector<LANES>
                    where
                        crate::$vector<LANES>: LanesAtMost32,
                    {
                        type Output = Self;

                        #[inline]
                        fn shl(self, rhs: Self) -> Self::Output {
                            // TODO there is probably a better way of doing this
                            if rhs.as_slice()
                                .iter()
                                .copied()
                                .any(invalid_shift_rhs)
                            {
                                panic!("attempt to shift left with overflow");
                            }
                            unsafe { crate::intrinsics::simd_shl(self, rhs) }
                        }
                    }
                }

                impl_ref_ops! {
                    impl<const LANES: usize> core::ops::Shl<$scalar> for crate::$vector<LANES>
                    where
                        crate::$vector<LANES>: LanesAtMost32,
                    {
                        type Output = Self;

                        #[inline]
                        fn shl(self, rhs: $scalar) -> Self::Output {
                            if invalid_shift_rhs(rhs) {
                                panic!("attempt to shift left with overflow");
                            }
                            let rhs = Self::splat(rhs);
                            unsafe { crate::intrinsics::simd_shl(self, rhs) }
                        }
                    }
                }


                impl_ref_ops! {
                    impl<const LANES: usize> core::ops::ShlAssign<Self> for crate::$vector<LANES>
                    where
                        crate::$vector<LANES>: LanesAtMost32,
                    {
                        #[inline]
                        fn shl_assign(&mut self, rhs: Self) {
                            *self = *self << rhs;
                        }
                    }
                }

                impl_ref_ops! {
                    impl<const LANES: usize> core::ops::ShlAssign<$scalar> for crate::$vector<LANES>
                    where
                        crate::$vector<LANES>: LanesAtMost32,
                    {
                        #[inline]
                        fn shl_assign(&mut self, rhs: $scalar) {
                            *self = *self << rhs;
                        }
                    }
                }

                impl_ref_ops! {
                    impl<const LANES: usize> core::ops::Shr<Self> for crate::$vector<LANES>
                    where
                        crate::$vector<LANES>: LanesAtMost32,
                    {
                        type Output = Self;

                        #[inline]
                        fn shr(self, rhs: Self) -> Self::Output {
                            // TODO there is probably a better way of doing this
                            if rhs.as_slice()
                                .iter()
                                .copied()
                                .any(invalid_shift_rhs)
                            {
                                panic!("attempt to shift with overflow");
                            }
                            unsafe { crate::intrinsics::simd_shr(self, rhs) }
                        }
                    }
                }

                impl_ref_ops! {
                    impl<const LANES: usize> core::ops::Shr<$scalar> for crate::$vector<LANES>
                    where
                        crate::$vector<LANES>: LanesAtMost32,
                    {
                        type Output = Self;

                        #[inline]
                        fn shr(self, rhs: $scalar) -> Self::Output {
                            if invalid_shift_rhs(rhs) {
                                panic!("attempt to shift with overflow");
                            }
                            let rhs = Self::splat(rhs);
                            unsafe { crate::intrinsics::simd_shr(self, rhs) }
                        }
                    }
                }


                impl_ref_ops! {
                    impl<const LANES: usize> core::ops::ShrAssign<Self> for crate::$vector<LANES>
                    where
                        crate::$vector<LANES>: LanesAtMost32,
                    {
                        #[inline]
                        fn shr_assign(&mut self, rhs: Self) {
                            *self = *self >> rhs;
                        }
                    }
                }

                impl_ref_ops! {
                    impl<const LANES: usize> core::ops::ShrAssign<$scalar> for crate::$vector<LANES>
                    where
                        crate::$vector<LANES>: LanesAtMost32,
                    {
                        #[inline]
                        fn shr_assign(&mut self, rhs: $scalar) {
                            *self = *self >> rhs;
                        }
                    }
                }
            )*
        )*
    };
}

/// Implements unsigned integer operators for the provided types.
macro_rules! impl_signed_int_ops {
    { $($scalar:ty => $($vector:ident),*;)* } => {
        impl_unsigned_int_ops! { $($scalar => $($vector),*;)* }
        $( // scalar
            $( // vector
                impl_op! { impl Neg for $vector, $scalar }
            )*
        )*
    };
}

impl_unsigned_int_ops! {
    u8 => SimdU8;
    u16 => SimdU16;
    u32 => SimdU32;
    u64 => SimdU64;
    u128 => SimdU128;
    usize => SimdUsize;
}

impl_signed_int_ops! {
    i8 => SimdI8;
    i16 => SimdI16;
    i32 => SimdI32;
    i64 => SimdI64;
    i128 => SimdI128;
    isize => SimdIsize;
}

impl_float_ops! {
    f32 => SimdF32;
    f64 => SimdF64;
}
