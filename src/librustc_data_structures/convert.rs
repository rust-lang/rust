use std::mem::transmute;

pub(crate) trait Convert<To> {
    fn convert(self) -> To;
    fn convert_ref(&self) -> &To;
    fn convert_mut_ref(&mut self) -> &mut To;
}
macro_rules! convert {
    ($from:ty, $to:ty) => {
        impl Convert<$to> for $from {
            #[inline(always)]
            fn convert(self) -> $to {
                unsafe { transmute(self) }
            }
            #[inline(always)]
            fn convert_ref(&self) -> &$to {
                unsafe { transmute(self) }
            }
            #[inline(always)]
            fn convert_mut_ref(&mut self) -> &mut $to {
                unsafe { transmute(self) }
            }
        }
        impl Convert<$from> for $to {
            #[inline(always)]
            fn convert(self) -> $from {
                unsafe { transmute(self) }
            }
            #[inline(always)]
            fn convert_ref(&self) -> &$from {
                unsafe { transmute(self) }
            }
            #[inline(always)]
            fn convert_mut_ref(&mut self) -> &mut $from {
                unsafe { transmute(self) }
            }
        }
    };
}
convert!(u128, [u64; 2]);
convert!(u128, [u32; 4]);
convert!(u128, [u16; 8]);
convert!(u128, [u8; 16]);
convert!([u64; 2], [u32; 4]);
convert!([u64; 2], [u16; 8]);
convert!([u64; 2], [u8; 16]);
convert!([u32; 4], [u16; 8]);
convert!([u32; 4], [u8; 16]);
convert!([u16; 8], [u8; 16]);
convert!(u64, [u32; 2]);
convert!(u64, [u16; 4]);
convert!(u64, [u8; 8]);
convert!([u32; 2], [u16; 4]);
convert!([u32; 2], [u8; 8]);
convert!(u32, [u16; 2]);
convert!(u32, [u8; 4]);
convert!([u16; 2], [u8; 4]);
convert!(u16, [u8; 2]);

convert!([f64; 2], [u8; 16]);
convert!([f32; 4], [u8; 16]);
convert!(f64, [u8; 8]);
convert!([f32; 2], [u8; 8]);
convert!(f32, [u8; 4]);



macro_rules! as_array {
    ($input:expr, $len:expr) => {{
        {
            #[inline]
            fn as_array<T>(slice: &[T]) -> &[T; $len] {
                assert_eq!(slice.len(), $len);
                unsafe {
                    &*(slice.as_ptr() as *const [_; $len])
                }
            }
            as_array($input)
        }
    }}
}
