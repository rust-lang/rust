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
        impl core::ops::$trait:ident<$rhs:ty> for $type:ty {
            type Output = $output:ty;

            $(#[$attrs:meta])*
            fn $fn:ident($self_tok:ident, $rhs_arg:ident: $rhs_arg_ty:ty) -> Self::Output $body:tt
        }
    } => {
        impl core::ops::$trait<$rhs> for $type {
            type Output = $output;

            $(#[$attrs])*
            fn $fn($self_tok, $rhs_arg: $rhs_arg_ty) -> Self::Output $body
        }

        impl core::ops::$trait<&'_ $rhs> for $type {
            type Output = <$type as core::ops::$trait<$rhs>>::Output;

            $(#[$attrs])*
            fn $fn($self_tok, $rhs_arg: &$rhs) -> Self::Output {
                core::ops::$trait::$fn($self_tok, *$rhs_arg)
            }
        }

        impl core::ops::$trait<$rhs> for &'_ $type {
            type Output = <$type as core::ops::$trait<$rhs>>::Output;

            $(#[$attrs])*
            fn $fn($self_tok, $rhs_arg: $rhs) -> Self::Output {
                core::ops::$trait::$fn(*$self_tok, $rhs_arg)
            }
        }

        impl core::ops::$trait<&'_ $rhs> for &'_ $type {
            type Output = <$type as core::ops::$trait<$rhs>>::Output;

            $(#[$attrs])*
            fn $fn($self_tok, $rhs_arg: &$rhs) -> Self::Output {
                core::ops::$trait::$fn(*$self_tok, *$rhs_arg)
            }
        }
    };

    // binary assignment op
    {
        impl core::ops::$trait:ident<$rhs:ty> for $type:ty {
            $(#[$attrs:meta])*
            fn $fn:ident(&mut $self_tok:ident, $rhs_arg:ident: $rhs_arg_ty:ty) $body:tt
        }
    } => {
        impl core::ops::$trait<$rhs> for $type {
            $(#[$attrs])*
            fn $fn(&mut $self_tok, $rhs_arg: $rhs_arg_ty) $body
        }

        impl core::ops::$trait<&'_ $rhs> for $type {
            $(#[$attrs])*
            fn $fn(&mut $self_tok, $rhs_arg: &$rhs_arg_ty) {
                core::ops::$trait::$fn($self_tok, *$rhs_arg)
            }
        }
    };

    // unary op
    {
        impl core::ops::$trait:ident for $type:ty {
            type Output = $output:ty;
            fn $fn:ident($self_tok:ident) -> Self::Output $body:tt
        }
    } => {
        impl core::ops::$trait for $type {
            type Output = $output;
            fn $fn($self_tok) -> Self::Output $body
        }

        impl core::ops::$trait for &'_ $type {
            type Output = <$type as core::ops::$trait>::Output;
            fn $fn($self_tok) -> Self::Output {
                core::ops::$trait::$fn(*$self_tok)
            }
        }
    }
}

/// Implements op traits for masks
macro_rules! impl_mask_element_ops {
    { $($mask:ty),* } => {
        $(
            impl_ref_ops! {
                impl core::ops::BitAnd<$mask> for $mask {
                    type Output = Self;
                    fn bitand(self, rhs: Self) -> Self::Output {
                        Self(self.0 & rhs.0)
                    }
                }
            }

            impl_ref_ops! {
                impl core::ops::BitAndAssign<$mask> for $mask {
                    fn bitand_assign(&mut self, rhs: Self) {
                        *self = *self & rhs;
                    }
                }
            }

            impl_ref_ops! {
                impl core::ops::BitOr<$mask> for $mask {
                    type Output = Self;
                    fn bitor(self, rhs: Self) -> Self::Output {
                        Self(self.0 | rhs.0)
                    }
                }
            }

            impl_ref_ops! {
                impl core::ops::BitOrAssign<$mask> for $mask {
                    fn bitor_assign(&mut self, rhs: Self) {
                        *self = *self | rhs;
                    }
                }
            }

            impl_ref_ops! {
                impl core::ops::BitXor<$mask> for $mask {
                    type Output = Self;
                    fn bitxor(self, rhs: Self) -> Self::Output {
                        Self(self.0 ^ rhs.0)
                    }
                }
            }

            impl_ref_ops! {
                impl core::ops::BitXorAssign<$mask> for $mask {
                    fn bitxor_assign(&mut self, rhs: Self) {
                        *self = *self ^ rhs;
                    }
                }
            }

            impl_ref_ops! {
                impl core::ops::Not for $mask {
                    type Output = Self;
                    fn not(self) -> Self::Output {
                        Self(!self.0)
                    }
                }
            }
        )*
    }
}

impl_mask_element_ops! {
    crate::masks::wide::m8,
    crate::masks::wide::m16,
    crate::masks::wide::m32,
    crate::masks::wide::m64,
    crate::masks::wide::m128,
    crate::masks::wide::msize
}

/// Automatically implements operators over vectors and scalars for a particular vector.
macro_rules! impl_op {
    { impl Add for $type:ty, $scalar:ty } => {
        impl_op! { @binary $type, $scalar, Add::add, AddAssign::add_assign, simd_add }
    };
    { impl Sub for $type:ty, $scalar:ty } => {
        impl_op! { @binary $type, $scalar, Sub::sub, SubAssign::sub_assign, simd_sub }
    };
    { impl Mul for $type:ty, $scalar:ty } => {
        impl_op! { @binary $type, $scalar, Mul::mul, MulAssign::mul_assign, simd_mul }
    };
    { impl Div for $type:ty, $scalar:ty } => {
        impl_op! { @binary $type, $scalar, Div::div, DivAssign::div_assign, simd_div }
    };
    { impl Rem for $type:ty, $scalar:ty } => {
        impl_op! { @binary $type, $scalar, Rem::rem, RemAssign::rem_assign, simd_rem }
    };
    { impl Shl for $type:ty, $scalar:ty } => {
        impl_op! { @binary $type, $scalar, Shl::shl, ShlAssign::shl_assign, simd_shl }
    };
    { impl Shr for $type:ty, $scalar:ty } => {
        impl_op! { @binary $type, $scalar, Shr::shr, ShrAssign::shr_assign, simd_shr }
    };
    { impl BitAnd for $type:ty, $scalar:ty } => {
        impl_op! { @binary $type, $scalar, BitAnd::bitand, BitAndAssign::bitand_assign, simd_and }
    };
    { impl BitOr for $type:ty, $scalar:ty } => {
        impl_op! { @binary $type, $scalar, BitOr::bitor, BitOrAssign::bitor_assign, simd_or }
    };
    { impl BitXor for $type:ty, $scalar:ty } => {
        impl_op! { @binary $type, $scalar, BitXor::bitxor, BitXorAssign::bitxor_assign, simd_xor }
    };

    { impl Not for $type:ty, $scalar:ty } => {
        impl_ref_ops! {
            impl core::ops::Not for $type {
                type Output = Self;
                fn not(self) -> Self::Output {
                    self ^ <$type>::splat(!<$scalar>::default())
                }
            }
        }
    };

    { impl Neg for $type:ty, $scalar:ty } => {
        impl_ref_ops! {
            impl core::ops::Neg for $type {
                type Output = Self;
                fn neg(self) -> Self::Output {
                    <$type>::splat(0) - self
                }
            }
        }
    };

    { impl Neg for $type:ty, $scalar:ty, @float } => {
        impl_ref_ops! {
            impl core::ops::Neg for $type {
                type Output = Self;
                fn neg(self) -> Self::Output {
                    // FIXME: Replace this with fneg intrinsic once available.
                    // https://github.com/rust-lang/stdsimd/issues/32
                    Self::from_bits(<$type>::splat(-0.0).to_bits() ^ self.to_bits())
                }
            }
        }
    };

    { impl Index for $type:ty, $scalar:ty } => {
        impl<I> core::ops::Index<I> for $type
        where
            I: core::slice::SliceIndex<[$scalar]>,
        {
            type Output = I::Output;
            fn index(&self, index: I) -> &Self::Output {
                let slice: &[_] = self.as_ref();
                &slice[index]
            }
        }

        impl<I> core::ops::IndexMut<I> for $type
        where
            I: core::slice::SliceIndex<[$scalar]>,
        {
            fn index_mut(&mut self, index: I) -> &mut Self::Output {
                let slice: &mut [_] = self.as_mut();
                &mut slice[index]
            }
        }
    };

    // generic binary op with assignment when output is `Self`
    { @binary $type:ty, $scalar:ty, $trait:ident :: $trait_fn:ident, $assign_trait:ident :: $assign_trait_fn:ident, $intrinsic:ident } => {
        impl_ref_ops! {
            impl core::ops::$trait<$type> for $type {
                type Output = $type;

                #[inline]
                fn $trait_fn(self, rhs: $type) -> Self::Output {
                    unsafe {
                        crate::intrinsics::$intrinsic(self, rhs)
                    }
                }
            }
        }

        impl_ref_ops! {
            impl core::ops::$trait<$scalar> for $type {
                type Output = $type;

                #[inline]
                fn $trait_fn(self, rhs: $scalar) -> Self::Output {
                    core::ops::$trait::$trait_fn(self, <$type>::splat(rhs))
                }
            }
        }

        impl_ref_ops! {
            impl core::ops::$trait<$type> for $scalar {
                type Output = $type;

                #[inline]
                fn $trait_fn(self, rhs: $type) -> Self::Output {
                    core::ops::$trait::$trait_fn(<$type>::splat(self), rhs)
                }
            }
        }

        impl_ref_ops! {
            impl core::ops::$assign_trait<$type> for $type {
                #[inline]
                fn $assign_trait_fn(&mut self, rhs: $type) {
                    unsafe {
                        *self = crate::intrinsics::$intrinsic(*self, rhs);
                    }
                }
            }
        }

        impl_ref_ops! {
            impl core::ops::$assign_trait<$scalar> for $type {
                #[inline]
                fn $assign_trait_fn(&mut self, rhs: $scalar) {
                    core::ops::$assign_trait::$assign_trait_fn(self, <$type>::splat(rhs));
                }
            }
        }
    };
}

/// Implements floating-point operators for the provided types.
macro_rules! impl_float_ops {
    { $($scalar:ty => $($vector:ty),*;)* } => {
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

/// Implements mask operators for the provided types.
macro_rules! impl_mask_ops {
    { $($scalar:ty => $($vector:ty),*;)* } => {
        $( // scalar
            $( // vector
                impl_op! { impl BitAnd for $vector, $scalar }
                impl_op! { impl BitOr  for $vector, $scalar }
                impl_op! { impl BitXor for $vector, $scalar }
                impl_op! { impl Not for $vector, $scalar }
                impl_op! { impl Index for $vector, $scalar }
            )*
        )*
    };
}

/// Implements unsigned integer operators for the provided types.
macro_rules! impl_unsigned_int_ops {
    { $($scalar:ty => $($vector:ty),*;)* } => {
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
                    impl core::ops::Div<$vector> for $vector {
                        type Output = Self;

                        #[inline]
                        fn div(self, rhs: $vector) -> Self::Output {
                            // TODO there is probably a better way of doing this
                            if AsRef::<[$scalar]>::as_ref(&rhs)
                                .iter()
                                .any(|x| *x == 0)
                            {
                                panic!("attempt to divide by zero");
                            }
                            unsafe { crate::intrinsics::simd_div(self, rhs) }
                        }
                    }
                }

                impl_ref_ops! {
                    impl core::ops::Div<$scalar> for $vector {
                        type Output = $vector;

                        #[inline]
                        fn div(self, rhs: $scalar) -> Self::Output {
                            if rhs == 0 {
                                panic!("attempt to divide by zero");
                            }
                            let rhs = Self::splat(rhs);
                            unsafe { crate::intrinsics::simd_div(self, rhs) }
                        }
                    }
                }

                impl_ref_ops! {
                    impl core::ops::Div<$vector> for $scalar {
                        type Output = $vector;

                        #[inline]
                        fn div(self, rhs: $vector) -> Self::Output {
                            <$vector>::splat(self) / rhs
                        }
                    }
                }

                impl_ref_ops! {
                    impl core::ops::DivAssign<$vector> for $vector {
                        #[inline]
                        fn div_assign(&mut self, rhs: Self) {
                            *self = *self / rhs;
                        }
                    }
                }

                impl_ref_ops! {
                    impl core::ops::DivAssign<$scalar> for $vector {
                        #[inline]
                        fn div_assign(&mut self, rhs: $scalar) {
                            *self = *self / rhs;
                        }
                    }
                }

                // remainder panics on zero divisor
                impl_ref_ops! {
                    impl core::ops::Rem<$vector> for $vector {
                        type Output = Self;

                        #[inline]
                        fn rem(self, rhs: $vector) -> Self::Output {
                            // TODO there is probably a better way of doing this
                            if AsRef::<[$scalar]>::as_ref(&rhs)
                                .iter()
                                .any(|x| *x == 0)
                            {
                                panic!("attempt to calculate the remainder with a divisor of zero");
                            }
                            unsafe { crate::intrinsics::simd_rem(self, rhs) }
                        }
                    }
                }

                impl_ref_ops! {
                    impl core::ops::Rem<$scalar> for $vector {
                        type Output = $vector;

                        #[inline]
                        fn rem(self, rhs: $scalar) -> Self::Output {
                            if rhs == 0 {
                                panic!("attempt to calculate the remainder with a divisor of zero");
                            }
                            let rhs = Self::splat(rhs);
                            unsafe { crate::intrinsics::simd_rem(self, rhs) }
                        }
                    }
                }

                impl_ref_ops! {
                    impl core::ops::Rem<$vector> for $scalar {
                        type Output = $vector;

                        #[inline]
                        fn rem(self, rhs: $vector) -> Self::Output {
                            <$vector>::splat(self) % rhs
                        }
                    }
                }

                impl_ref_ops! {
                    impl core::ops::RemAssign<$vector> for $vector {
                        #[inline]
                        fn rem_assign(&mut self, rhs: Self) {
                            *self = *self % rhs;
                        }
                    }
                }

                impl_ref_ops! {
                    impl core::ops::RemAssign<$scalar> for $vector {
                        #[inline]
                        fn rem_assign(&mut self, rhs: $scalar) {
                            *self = *self % rhs;
                        }
                    }
                }

                // shifts panic on overflow
                impl_ref_ops! {
                    impl core::ops::Shl<$vector> for $vector {
                        type Output = Self;

                        #[inline]
                        fn shl(self, rhs: $vector) -> Self::Output {
                            // TODO there is probably a better way of doing this
                            if AsRef::<[$scalar]>::as_ref(&rhs)
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
                    impl core::ops::Shl<$scalar> for $vector {
                        type Output = $vector;

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
                    impl core::ops::ShlAssign<$vector> for $vector {
                        #[inline]
                        fn shl_assign(&mut self, rhs: Self) {
                            *self = *self << rhs;
                        }
                    }
                }

                impl_ref_ops! {
                    impl core::ops::ShlAssign<$scalar> for $vector {
                        #[inline]
                        fn shl_assign(&mut self, rhs: $scalar) {
                            *self = *self << rhs;
                        }
                    }
                }

                impl_ref_ops! {
                    impl core::ops::Shr<$vector> for $vector {
                        type Output = Self;

                        #[inline]
                        fn shr(self, rhs: $vector) -> Self::Output {
                            // TODO there is probably a better way of doing this
                            if AsRef::<[$scalar]>::as_ref(&rhs)
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
                    impl core::ops::Shr<$scalar> for $vector {
                        type Output = $vector;

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
                    impl core::ops::ShrAssign<$vector> for $vector {
                        #[inline]
                        fn shr_assign(&mut self, rhs: Self) {
                            *self = *self >> rhs;
                        }
                    }
                }

                impl_ref_ops! {
                    impl core::ops::ShrAssign<$scalar> for $vector {
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
    { $($scalar:ty => $($vector:ty),*;)* } => {
        impl_unsigned_int_ops! { $($scalar => $($vector),*;)* }
        $( // scalar
            $( // vector
                impl_op! { impl Neg for $vector, $scalar }
            )*
        )*
    };
}

impl_unsigned_int_ops! {
    u8    => crate::u8x8,    crate::u8x16,   crate::u8x32,   crate::u8x64;
    u16   => crate::u16x4,   crate::u16x8,   crate::u16x16,  crate::u16x32;
    u32   => crate::u32x2,   crate::u32x4,   crate::u32x8,   crate::u32x16;
    u64   => crate::u64x2,   crate::u64x4,   crate::u64x8;
    u128  => crate::u128x2,  crate::u128x4;
    usize => crate::usizex2, crate::usizex4, crate::usizex8;
}

impl_signed_int_ops! {
    i8    => crate::i8x8,    crate::i8x16,   crate::i8x32,   crate::i8x64;
    i16   => crate::i16x4,   crate::i16x8,   crate::i16x16,  crate::i16x32;
    i32   => crate::i32x2,   crate::i32x4,   crate::i32x8,   crate::i32x16;
    i64   => crate::i64x2,   crate::i64x4,   crate::i64x8;
    i128  => crate::i128x2,  crate::i128x4;
    isize => crate::isizex2, crate::isizex4, crate::isizex8;
}

impl_float_ops! {
    f32 => crate::f32x2, crate::f32x4, crate::f32x8, crate::f32x16;
    f64 => crate::f64x2, crate::f64x4, crate::f64x8;
}

impl_mask_ops! {
    crate::masks::wide::m8    => crate::masks::wide::m8x8,    crate::masks::wide::m8x16,   crate::masks::wide::m8x32,   crate::masks::wide::m8x64;
    crate::masks::wide::m16   => crate::masks::wide::m16x4,   crate::masks::wide::m16x8,   crate::masks::wide::m16x16,  crate::masks::wide::m16x32;
    crate::masks::wide::m32   => crate::masks::wide::m32x2,   crate::masks::wide::m32x4,   crate::masks::wide::m32x8,   crate::masks::wide::m32x16;
    crate::masks::wide::m64   => crate::masks::wide::m64x2,   crate::masks::wide::m64x4,   crate::masks::wide::m64x8;
    crate::masks::wide::m128  => crate::masks::wide::m128x2,  crate::masks::wide::m128x4;
    crate::masks::wide::msize => crate::masks::wide::msizex2, crate::masks::wide::msizex4, crate::masks::wide::msizex8;
}
