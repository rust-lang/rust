mod private {
    /// This trait being unreachable from outside the crate
    /// prevents other implementations of the `FloatToInt` trait,
    /// which allows potentially adding more trait methods after the trait is `#[stable]`.
    #[unstable(feature = "convert_float_to_int", issue = "67057")]
    pub trait Sealed {}
}

/// Supporting trait for inherent methods of `f32` and `f64` such as `round_unchecked_to`.
/// Typically doesn’t need to be used directly.
#[unstable(feature = "convert_float_to_int", issue = "67057")]
pub trait FloatToInt<Int>: private::Sealed + Sized {
    #[cfg(not(bootstrap))]
    #[unstable(feature = "float_approx_unchecked_to", issue = "67058")]
    #[doc(hidden)]
    unsafe fn approx_unchecked(self) -> Int;
}

macro_rules! impl_float_to_int {
    ( $Float: ident => $( $Int: ident )+ ) => {
        #[unstable(feature = "convert_float_to_int", issue = "67057")]
        impl private::Sealed for $Float {}
        $(
            #[unstable(feature = "convert_float_to_int", issue = "67057")]
            impl FloatToInt<$Int> for $Float {
                #[cfg(not(bootstrap))]
                #[doc(hidden)]
                #[inline]
                unsafe fn approx_unchecked(self) -> $Int {
                    crate::intrinsics::float_to_int_approx_unchecked(self)
                }
            }
        )+
    }
}

impl_float_to_int!(f32 => u8 u16 u32 u64 u128 usize i8 i16 i32 i64 i128 isize);
impl_float_to_int!(f64 => u8 u16 u32 u64 u128 usize i8 i16 i32 i64 i128 isize);
