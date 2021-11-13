use crate::simd::intrinsics;
use crate::simd::{LaneCount, Simd, SimdElement, SupportedLaneCount};

impl<I, T, const LANES: usize> core::ops::Index<I> for Simd<T, LANES>
where
    T: SimdElement,
    LaneCount<LANES>: SupportedLaneCount,
    I: core::slice::SliceIndex<[T]>,
{
    type Output = I::Output;
    fn index(&self, index: I) -> &Self::Output {
        &self.as_array()[index]
    }
}

impl<I, T, const LANES: usize> core::ops::IndexMut<I> for Simd<T, LANES>
where
    T: SimdElement,
    LaneCount<LANES>: SupportedLaneCount,
    I: core::slice::SliceIndex<[T]>,
{
    fn index_mut(&mut self, index: I) -> &mut Self::Output {
        &mut self.as_mut_array()[index]
    }
}

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
            LaneCount<$lanes2:ident>: SupportedLaneCount,
        {
            type Output = $output:ty;

            $(#[$attrs:meta])*
            fn $fn:ident($self_tok:ident, $rhs_arg:ident: $rhs_arg_ty:ty) -> Self::Output $body:tt
        }
    } => {
        impl<const $lanes: usize> core::ops::$trait<$rhs> for $type
        where
            LaneCount<$lanes2>: SupportedLaneCount,
        {
            type Output = $output;

            $(#[$attrs])*
            fn $fn($self_tok, $rhs_arg: $rhs_arg_ty) -> Self::Output $body
        }

        impl<const $lanes: usize> core::ops::$trait<&'_ $rhs> for $type
        where
            LaneCount<$lanes2>: SupportedLaneCount,
        {
            type Output = <$type as core::ops::$trait<$rhs>>::Output;

            $(#[$attrs])*
            fn $fn($self_tok, $rhs_arg: &$rhs) -> Self::Output {
                core::ops::$trait::$fn($self_tok, *$rhs_arg)
            }
        }

        impl<const $lanes: usize> core::ops::$trait<$rhs> for &'_ $type
        where
            LaneCount<$lanes2>: SupportedLaneCount,
        {
            type Output = <$type as core::ops::$trait<$rhs>>::Output;

            $(#[$attrs])*
            fn $fn($self_tok, $rhs_arg: $rhs) -> Self::Output {
                core::ops::$trait::$fn(*$self_tok, $rhs_arg)
            }
        }

        impl<const $lanes: usize> core::ops::$trait<&'_ $rhs> for &'_ $type
        where
            LaneCount<$lanes2>: SupportedLaneCount,
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
            LaneCount<$lanes2:ident>: SupportedLaneCount,
        {
            $(#[$attrs:meta])*
            fn $fn:ident(&mut $self_tok:ident, $rhs_arg:ident: $rhs_arg_ty:ty) $body:tt
        }
    } => {
        impl<const $lanes: usize> core::ops::$trait<$rhs> for $type
        where
            LaneCount<$lanes2>: SupportedLaneCount,
        {
            $(#[$attrs])*
            fn $fn(&mut $self_tok, $rhs_arg: $rhs_arg_ty) $body
        }

        impl<const $lanes: usize> core::ops::$trait<&'_ $rhs> for $type
        where
            LaneCount<$lanes2>: SupportedLaneCount,
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
            LaneCount<$lanes2:ident>: SupportedLaneCount,
        {
            type Output = $output:ty;
            fn $fn:ident($self_tok:ident) -> Self::Output $body:tt
        }
    } => {
        impl<const $lanes: usize> core::ops::$trait for $type
        where
            LaneCount<$lanes2>: SupportedLaneCount,
        {
            type Output = $output;
            fn $fn($self_tok) -> Self::Output $body
        }

        impl<const $lanes: usize> core::ops::$trait for &'_ $type
        where
            LaneCount<$lanes2>: SupportedLaneCount,
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
    { impl Add for $scalar:ty } => {
        impl_op! { @binary $scalar, Add::add, AddAssign::add_assign, simd_add }
    };
    { impl Sub for $scalar:ty } => {
        impl_op! { @binary $scalar, Sub::sub, SubAssign::sub_assign, simd_sub }
    };
    { impl Mul for $scalar:ty } => {
        impl_op! { @binary $scalar, Mul::mul, MulAssign::mul_assign, simd_mul }
    };
    { impl Div for $scalar:ty } => {
        impl_op! { @binary $scalar, Div::div, DivAssign::div_assign, simd_div }
    };
    { impl Rem for $scalar:ty } => {
        impl_op! { @binary $scalar, Rem::rem, RemAssign::rem_assign, simd_rem }
    };
    { impl Shl for $scalar:ty } => {
        impl_op! { @binary $scalar, Shl::shl, ShlAssign::shl_assign, simd_shl }
    };
    { impl Shr for $scalar:ty } => {
        impl_op! { @binary $scalar, Shr::shr, ShrAssign::shr_assign, simd_shr }
    };
    { impl BitAnd for $scalar:ty } => {
        impl_op! { @binary $scalar, BitAnd::bitand, BitAndAssign::bitand_assign, simd_and }
    };
    { impl BitOr for $scalar:ty } => {
        impl_op! { @binary $scalar, BitOr::bitor, BitOrAssign::bitor_assign, simd_or }
    };
    { impl BitXor for $scalar:ty } => {
        impl_op! { @binary $scalar, BitXor::bitxor, BitXorAssign::bitxor_assign, simd_xor }
    };

    { impl Not for $scalar:ty } => {
        impl_ref_ops! {
            impl<const LANES: usize> core::ops::Not for Simd<$scalar, LANES>
            where
                LaneCount<LANES>: SupportedLaneCount,
            {
                type Output = Self;
                fn not(self) -> Self::Output {
                    self ^ Self::splat(!<$scalar>::default())
                }
            }
        }
    };

    { impl Neg for $scalar:ty } => {
        impl_ref_ops! {
            impl<const LANES: usize> core::ops::Neg for Simd<$scalar, LANES>
            where
                LaneCount<LANES>: SupportedLaneCount,
            {
                type Output = Self;
                fn neg(self) -> Self::Output {
                    unsafe { intrinsics::simd_neg(self) }
                }
            }
        }
    };

    // generic binary op with assignment when output is `Self`
    { @binary $scalar:ty, $trait:ident :: $trait_fn:ident, $assign_trait:ident :: $assign_trait_fn:ident, $intrinsic:ident } => {
        impl_ref_ops! {
            impl<const LANES: usize> core::ops::$trait<Self> for Simd<$scalar, LANES>
            where
                LaneCount<LANES>: SupportedLaneCount,
            {
                type Output = Self;

                #[inline]
                fn $trait_fn(self, rhs: Self) -> Self::Output {
                    unsafe {
                        intrinsics::$intrinsic(self, rhs)
                    }
                }
            }
        }

        impl_ref_ops! {
            impl<const LANES: usize> core::ops::$trait<$scalar> for Simd<$scalar, LANES>
            where
                LaneCount<LANES>: SupportedLaneCount,
            {
                type Output = Self;

                #[inline]
                fn $trait_fn(self, rhs: $scalar) -> Self::Output {
                    core::ops::$trait::$trait_fn(self, Self::splat(rhs))
                }
            }
        }

        impl_ref_ops! {
            impl<const LANES: usize> core::ops::$trait<Simd<$scalar, LANES>> for $scalar
            where
                LaneCount<LANES>: SupportedLaneCount,
            {
                type Output = Simd<$scalar, LANES>;

                #[inline]
                fn $trait_fn(self, rhs: Simd<$scalar, LANES>) -> Self::Output {
                    core::ops::$trait::$trait_fn(Simd::splat(self), rhs)
                }
            }
        }

        impl_ref_ops! {
            impl<const LANES: usize> core::ops::$assign_trait<Self> for Simd<$scalar, LANES>
            where
                LaneCount<LANES>: SupportedLaneCount,
            {
                #[inline]
                fn $assign_trait_fn(&mut self, rhs: Self) {
                    unsafe {
                        *self = intrinsics::$intrinsic(*self, rhs);
                    }
                }
            }
        }

        impl_ref_ops! {
            impl<const LANES: usize> core::ops::$assign_trait<$scalar> for Simd<$scalar, LANES>
            where
                LaneCount<LANES>: SupportedLaneCount,
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
    { $($scalar:ty),* } => {
        $(
            impl_op! { impl Add for $scalar }
            impl_op! { impl Sub for $scalar }
            impl_op! { impl Mul for $scalar }
            impl_op! { impl Div for $scalar }
            impl_op! { impl Rem for $scalar }
            impl_op! { impl Neg for $scalar }
        )*
    };
}

/// Implements unsigned integer operators for the provided types.
macro_rules! impl_unsigned_int_ops {
    { $($scalar:ty),* } => {
        $(
            impl_op! { impl Add for $scalar }
            impl_op! { impl Sub for $scalar }
            impl_op! { impl Mul for $scalar }
            impl_op! { impl BitAnd for $scalar }
            impl_op! { impl BitOr  for $scalar }
            impl_op! { impl BitXor for $scalar }
            impl_op! { impl Not for $scalar }

            // Integers panic on divide by 0
            impl_ref_ops! {
                impl<const LANES: usize> core::ops::Div<Self> for Simd<$scalar, LANES>
                where
                    LaneCount<LANES>: SupportedLaneCount,
                {
                    type Output = Self;

                    #[inline]
                    fn div(self, rhs: Self) -> Self::Output {
                        if rhs.as_array()
                            .iter()
                            .any(|x| *x == 0)
                        {
                            panic!("attempt to divide by zero");
                        }

                        // Guards for div(MIN, -1),
                        // this check only applies to signed ints
                        if <$scalar>::MIN != 0 && self.as_array().iter()
                                .zip(rhs.as_array().iter())
                                .any(|(x,y)| *x == <$scalar>::MIN && *y == -1 as _) {
                            panic!("attempt to divide with overflow");
                        }
                        unsafe { intrinsics::simd_div(self, rhs) }
                    }
                }
            }

            impl_ref_ops! {
                impl<const LANES: usize> core::ops::Div<$scalar> for Simd<$scalar, LANES>
                where
                    LaneCount<LANES>: SupportedLaneCount,
                {
                    type Output = Self;

                    #[inline]
                    fn div(self, rhs: $scalar) -> Self::Output {
                        if rhs == 0 {
                            panic!("attempt to divide by zero");
                        }
                        if <$scalar>::MIN != 0 &&
                            self.as_array().iter().any(|x| *x == <$scalar>::MIN) &&
                            rhs == -1 as _ {
                                panic!("attempt to divide with overflow");
                        }
                        let rhs = Self::splat(rhs);
                        unsafe { intrinsics::simd_div(self, rhs) }
                    }
                }
            }

            impl_ref_ops! {
                impl<const LANES: usize> core::ops::Div<Simd<$scalar, LANES>> for $scalar
                where
                    LaneCount<LANES>: SupportedLaneCount,
                {
                    type Output = Simd<$scalar, LANES>;

                    #[inline]
                    fn div(self, rhs: Simd<$scalar, LANES>) -> Self::Output {
                        Simd::splat(self) / rhs
                    }
                }
            }

            impl_ref_ops! {
                impl<const LANES: usize> core::ops::DivAssign<Self> for Simd<$scalar, LANES>
                where
                    LaneCount<LANES>: SupportedLaneCount,
                {
                    #[inline]
                    fn div_assign(&mut self, rhs: Self) {
                        *self = *self / rhs;
                    }
                }
            }

            impl_ref_ops! {
                impl<const LANES: usize> core::ops::DivAssign<$scalar> for Simd<$scalar, LANES>
                where
                    LaneCount<LANES>: SupportedLaneCount,
                {
                    #[inline]
                    fn div_assign(&mut self, rhs: $scalar) {
                        *self = *self / rhs;
                    }
                }
            }

            // remainder panics on zero divisor
            impl_ref_ops! {
                impl<const LANES: usize> core::ops::Rem<Self> for Simd<$scalar, LANES>
                where
                    LaneCount<LANES>: SupportedLaneCount,
                {
                    type Output = Self;

                    #[inline]
                    fn rem(self, rhs: Self) -> Self::Output {
                        if rhs.as_array()
                            .iter()
                            .any(|x| *x == 0)
                        {
                            panic!("attempt to calculate the remainder with a divisor of zero");
                        }

                        // Guards for rem(MIN, -1)
                        // this branch applies the check only to signed ints
                        if <$scalar>::MIN != 0 && self.as_array().iter()
                                .zip(rhs.as_array().iter())
                                .any(|(x,y)| *x == <$scalar>::MIN && *y == -1 as _) {
                            panic!("attempt to calculate the remainder with overflow");
                        }
                        unsafe { intrinsics::simd_rem(self, rhs) }
                    }
                }
            }

            impl_ref_ops! {
                impl<const LANES: usize> core::ops::Rem<$scalar> for Simd<$scalar, LANES>
                where
                    LaneCount<LANES>: SupportedLaneCount,
                {
                    type Output = Self;

                    #[inline]
                    fn rem(self, rhs: $scalar) -> Self::Output {
                        if rhs == 0 {
                            panic!("attempt to calculate the remainder with a divisor of zero");
                        }
                        if <$scalar>::MIN != 0 &&
                            self.as_array().iter().any(|x| *x == <$scalar>::MIN) &&
                            rhs == -1 as _ {
                                panic!("attempt to calculate the remainder with overflow");
                        }
                        let rhs = Self::splat(rhs);
                        unsafe { intrinsics::simd_rem(self, rhs) }
                    }
                }
            }

            impl_ref_ops! {
                impl<const LANES: usize> core::ops::Rem<Simd<$scalar, LANES>> for $scalar
                where
                    LaneCount<LANES>: SupportedLaneCount,
                {
                    type Output = Simd<$scalar, LANES>;

                    #[inline]
                    fn rem(self, rhs: Simd<$scalar, LANES>) -> Self::Output {
                        Simd::splat(self) % rhs
                    }
                }
            }

            impl_ref_ops! {
                impl<const LANES: usize> core::ops::RemAssign<Self> for Simd<$scalar, LANES>
                where
                    LaneCount<LANES>: SupportedLaneCount,
                {
                    #[inline]
                    fn rem_assign(&mut self, rhs: Self) {
                        *self = *self % rhs;
                    }
                }
            }

            impl_ref_ops! {
                impl<const LANES: usize> core::ops::RemAssign<$scalar> for Simd<$scalar, LANES>
                where
                    LaneCount<LANES>: SupportedLaneCount,
                {
                    #[inline]
                    fn rem_assign(&mut self, rhs: $scalar) {
                        *self = *self % rhs;
                    }
                }
            }

            // shifts panic on overflow
            impl_ref_ops! {
                impl<const LANES: usize> core::ops::Shl<Self> for Simd<$scalar, LANES>
                where
                    LaneCount<LANES>: SupportedLaneCount,
                {
                    type Output = Self;

                    #[inline]
                    fn shl(self, rhs: Self) -> Self::Output {
                        // TODO there is probably a better way of doing this
                        if rhs.as_array()
                            .iter()
                            .copied()
                            .any(invalid_shift_rhs)
                        {
                            panic!("attempt to shift left with overflow");
                        }
                        unsafe { intrinsics::simd_shl(self, rhs) }
                    }
                }
            }

            impl_ref_ops! {
                impl<const LANES: usize> core::ops::Shl<$scalar> for Simd<$scalar, LANES>
                where
                    LaneCount<LANES>: SupportedLaneCount,
                {
                    type Output = Self;

                    #[inline]
                    fn shl(self, rhs: $scalar) -> Self::Output {
                        if invalid_shift_rhs(rhs) {
                            panic!("attempt to shift left with overflow");
                        }
                        let rhs = Self::splat(rhs);
                        unsafe { intrinsics::simd_shl(self, rhs) }
                    }
                }
            }


            impl_ref_ops! {
                impl<const LANES: usize> core::ops::ShlAssign<Self> for Simd<$scalar, LANES>
                where
                    LaneCount<LANES>: SupportedLaneCount,
                {
                    #[inline]
                    fn shl_assign(&mut self, rhs: Self) {
                        *self = *self << rhs;
                    }
                }
            }

            impl_ref_ops! {
                impl<const LANES: usize> core::ops::ShlAssign<$scalar> for Simd<$scalar, LANES>
                where
                    LaneCount<LANES>: SupportedLaneCount,
                {
                    #[inline]
                    fn shl_assign(&mut self, rhs: $scalar) {
                        *self = *self << rhs;
                    }
                }
            }

            impl_ref_ops! {
                impl<const LANES: usize> core::ops::Shr<Self> for Simd<$scalar, LANES>
                where
                    LaneCount<LANES>: SupportedLaneCount,
                {
                    type Output = Self;

                    #[inline]
                    fn shr(self, rhs: Self) -> Self::Output {
                        // TODO there is probably a better way of doing this
                        if rhs.as_array()
                            .iter()
                            .copied()
                            .any(invalid_shift_rhs)
                        {
                            panic!("attempt to shift with overflow");
                        }
                        unsafe { intrinsics::simd_shr(self, rhs) }
                    }
                }
            }

            impl_ref_ops! {
                impl<const LANES: usize> core::ops::Shr<$scalar> for Simd<$scalar, LANES>
                where
                    LaneCount<LANES>: SupportedLaneCount,
                {
                    type Output = Self;

                    #[inline]
                    fn shr(self, rhs: $scalar) -> Self::Output {
                        if invalid_shift_rhs(rhs) {
                            panic!("attempt to shift with overflow");
                        }
                        let rhs = Self::splat(rhs);
                        unsafe { intrinsics::simd_shr(self, rhs) }
                    }
                }
            }


            impl_ref_ops! {
                impl<const LANES: usize> core::ops::ShrAssign<Self> for Simd<$scalar, LANES>
                where
                    LaneCount<LANES>: SupportedLaneCount,
                {
                    #[inline]
                    fn shr_assign(&mut self, rhs: Self) {
                        *self = *self >> rhs;
                    }
                }
            }

            impl_ref_ops! {
                impl<const LANES: usize> core::ops::ShrAssign<$scalar> for Simd<$scalar, LANES>
                where
                    LaneCount<LANES>: SupportedLaneCount,
                {
                    #[inline]
                    fn shr_assign(&mut self, rhs: $scalar) {
                        *self = *self >> rhs;
                    }
                }
            }
        )*
    };
}

/// Implements unsigned integer operators for the provided types.
macro_rules! impl_signed_int_ops {
    { $($scalar:ty),* } => {
        impl_unsigned_int_ops! { $($scalar),* }
        $( // scalar
            impl_op! { impl Neg for $scalar }
        )*
    };
}

impl_unsigned_int_ops! { u8, u16, u32, u64, usize }
impl_signed_int_ops! { i8, i16, i32, i64, isize }
impl_float_ops! { f32, f64 }
