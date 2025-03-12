use crate::mem::MaybeUninit;
use crate::slice;

/// A minimal buffer implementation containing elements of type
/// `MaybeUninit<u8>`.
#[unstable(feature = "int_format_into", issue = "138215")]
#[derive(Debug)]
pub struct NumBuffer<const BUF_SIZE: usize> {
    /// An array of elements of type `MaybeUninit<u8>`.
    ///
    /// An alternative to `contents.len()` is `BUF_SIZE`.
    pub contents: [MaybeUninit<u8>; BUF_SIZE],
}

#[unstable(feature = "int_format_into", issue = "138215")]
impl<const BUF_SIZE: usize> NumBuffer<BUF_SIZE> {
    /// Initializes `contents` as an uninitialized array of `MaybeUninit<u8>`.
    #[unstable(feature = "int_format_into", issue = "138215")]
    pub fn new() -> Self {
        NumBuffer { contents: [MaybeUninit::<u8>::uninit(); BUF_SIZE] }
    }
}

const DEC_DIGITS_LUT: &[u8; 200] = b"\
        0001020304050607080910111213141516171819\
        2021222324252627282930313233343536373839\
        4041424344454647484950515253545556575859\
        6061626364656667686970717273747576777879\
        8081828384858687888990919293949596979899";

const NEGATIVE_SIGN: &[u8; 1] = b"-";

/// Writes the negative sign into the buffer.
/// Must be called only in the case where the number is negative.
fn raw_write_sign_into<const BUF_SIZE: usize>(
    buf: &mut NumBuffer<BUF_SIZE>,
    start_offset: usize,
) -> usize {
    let mut offset = start_offset;

    // SAFETY: All of the decimals (with the sign) fit in buf, since it now is size-checked
    // and the if condition ensures (at least) that the sign can be added.
    unsafe { core::hint::assert_unchecked(offset >= 1) }

    // SAFETY: The offset counts down from its initial value BUF_SIZE
    // without underflow due to the previous precondition.
    unsafe { core::hint::assert_unchecked(offset <= BUF_SIZE) }

    // Setting sign for the negative number
    offset -= 1;
    buf.contents[offset].write(NEGATIVE_SIGN[0]);

    offset
}

/// Exports the stringified version of the buffer.
fn raw_extract_as_str<const BUF_SIZE: usize>(buf: &NumBuffer<BUF_SIZE>, offset: usize) -> &str {
    // SAFETY: All contents of `buf` since `offset` is set.
    let written = unsafe { buf.contents.get_unchecked(offset..) };

    // SAFETY: Writes use ASCII from the lookup table
    // (and `NEGATIVE_SIGN` in case of negative numbers) exclusively.
    let as_str = unsafe {
        str::from_utf8_unchecked(slice::from_raw_parts(
            MaybeUninit::slice_as_ptr(written),
            written.len(),
        ))
    };

    as_str
}

/// Macro to write ascii digits into the buffer, towards the end.
macro_rules! raw_write_digits_into {
    ($T:ty, $BUF_SIZE:expr) => {
        |target_var: $T, buf: &mut NumBuffer<{ $BUF_SIZE }>, start_offset: usize| -> usize {
            let mut offset = start_offset;

            // Consume the least-significant decimals from a working copy.
            let mut remain: $T = target_var;

            // Format per four digits from the lookup table.
            // Four digits need a 16-bit $unsigned or wider.
            while size_of::<$T>() > 1
                && remain
                    > 999.try_into().expect("branch is not hit for types that cannot fit 999 (u8)")
            {
                // SAFETY: All of the decimals fit in buf, since it now is size-checked
                // and the while condition ensures at least 4 more decimals.
                unsafe { core::hint::assert_unchecked(offset >= 4) }
                // SAFETY: The offset counts down from its initial value $BUF_SIZE
                // without underflow due to the previous precondition.
                unsafe { core::hint::assert_unchecked(offset <= $BUF_SIZE) }
                offset -= 4;

                // pull two pairs
                let scale: $T = 1_00_00
                    .try_into()
                    .expect("branch is not hit for types that cannot fit 1E4 (u8)");

                let quad = remain % scale;
                remain /= scale;
                let pair1 = (quad / 100) as usize;
                let pair2 = (quad % 100) as usize;
                buf.contents[offset + 0].write(DEC_DIGITS_LUT[pair1 * 2 + 0]);
                buf.contents[offset + 1].write(DEC_DIGITS_LUT[pair1 * 2 + 1]);
                buf.contents[offset + 2].write(DEC_DIGITS_LUT[pair2 * 2 + 0]);
                buf.contents[offset + 3].write(DEC_DIGITS_LUT[pair2 * 2 + 1]);
            }

            // Format per two digits from the lookup table.
            if remain > 9 {
                // SAFETY: All of the decimals fit in buf, since it now is size-checked
                // and the while condition ensures at least 2 more decimals.
                unsafe { core::hint::assert_unchecked(offset >= 2) }
                // SAFETY: The offset counts down from its initial value $BUF_SIZE
                // without underflow due to the previous precondition.
                unsafe { core::hint::assert_unchecked(offset <= $BUF_SIZE) }
                offset -= 2;

                let pair = (remain % 100) as usize;
                remain /= 100;
                buf.contents[offset + 0].write(DEC_DIGITS_LUT[pair * 2 + 0]);
                buf.contents[offset + 1].write(DEC_DIGITS_LUT[pair * 2 + 1]);
            }

            // Format the last remaining digit, if any.
            if remain != 0 || target_var == 0 {
                // SAFETY: All of the decimals fit in buf, since it now is size-checked
                // and the if condition ensures (at least) 1 more decimals.
                unsafe { core::hint::assert_unchecked(offset >= 1) }
                // SAFETY: The offset counts down from its initial value $BUF_SIZE
                // without underflow due to the previous precondition.
                unsafe { core::hint::assert_unchecked(offset <= $BUF_SIZE) }
                offset -= 1;

                // Either the compiler sees that remain < 10, or it prevents
                // a boundary check up next.
                let last = (remain & 15) as usize;
                buf.contents[offset].write(DEC_DIGITS_LUT[last * 2 + 1]);
                // not used: remain = 0;
            }

            offset
        }
    };
}

macro_rules! macro_impl_format_into {
    (signed; $(($SignedT:ty, $UnsignedT:ty))*) => {
        $(
            #[unstable(feature = "int_format_into", issue = "138215")]
            impl $SignedT {
                /// Allows users to write an integer (in signed decimal format) into a variable `buf` of
                /// type [`NumBuffer`] that is passed by the caller by mutable reference.
                ///
                /// This function panics if `buf` does not have enough size to store
                /// the signed decimal version of the number.
                ///
                /// # Examples
                /// ```
                /// #![feature(int_format_into)]
                /// use core::num::NumBuffer;
                ///
                #[doc = concat!("let n = -32", stringify!($SignedT), ";")]
                /// let mut buf = NumBuffer::<3>::new();
                /// assert_eq!(n.format_into(&mut buf), "-32");
                ///
                #[doc = concat!("let n2 = ", stringify!($SignedT::MIN), ";")]
                /// let mut buf2 = NumBuffer::<40>::new();
                #[doc = concat!("assert_eq!(n2.format_into(&mut buf2), ", stringify!($SignedT::MIN), ".to_string());")]
                ///
                #[doc = concat!("let n3 = ", stringify!($SignedT::MAX), ";")]
                /// let mut buf3 = NumBuffer::<40>::new();
                #[doc = concat!("assert_eq!(n3.format_into(&mut buf3), ", stringify!($SignedT::MAX), ".to_string());")]
                /// ```
                ///
                pub fn format_into<const BUF_SIZE: usize>(self, buf: &mut crate::num::NumBuffer<BUF_SIZE>) -> &str {
                    // counting space for negative sign too, if `self` is negative
                    let decimal_string_size: usize = if self < 0 {
                        self.unsigned_abs().ilog(10) as usize + 1 + 1
                    } else if self == 0 {
                        1
                    } else {
                        self.ilog(10) as usize + 1
                    };

                    // `buf` must have minimum size to store the decimal string version.
                    // BUF_SIZE is the size of the buffer.
                    if BUF_SIZE < decimal_string_size {
                        panic!("Not enough buffer size to format into!");
                    }

                    // Count the number of bytes in `buf` that are not initialized.
                    let mut offset = BUF_SIZE;

                    offset = raw_write_digits_into!($UnsignedT, BUF_SIZE)(self.unsigned_abs(), buf, offset);

                    if self < 0 {
                        offset = raw_write_sign_into(buf, offset);
                    }

                    raw_extract_as_str(buf, offset)
                }
            }
        )*
    };

    (unsigned; $($UnsignedT:ty)*) => {
        $(
            #[unstable(feature = "int_format_into", issue = "138215")]
            impl $UnsignedT {
                /// Allows users to write an integer (in signed decimal format) into a variable `buf` of
                /// type [`NumBuffer`] that is passed by the caller by mutable reference.
                ///
                /// This function panics if `buf` does not have enough size to store
                /// the signed decimal version of the number.
                ///
                /// # Examples
                /// ```
                /// #![feature(int_format_into)]
                /// use core::num::NumBuffer;
                ///
                #[doc = concat!("let n = 32", stringify!($UnsignedT), ";")]
                /// let mut buf = NumBuffer::<3>::new();
                /// assert_eq!(n.format_into(&mut buf), "32");
                ///
                #[doc = concat!("let n2 = ", stringify!($UnsignedT::MAX), ";")]
                /// let mut buf2 = NumBuffer::<40>::new();
                #[doc = concat!("assert_eq!(n2.format_into(&mut buf2), ", stringify!($UnsignedT::MAX), ".to_string());")]
                /// ```
                ///
                pub fn format_into<const BUF_SIZE: usize>(self, buf: &mut crate::num::NumBuffer<BUF_SIZE>) -> &str {
                    // counting space for negative sign too, if `self` is negative
                    let decimal_string_size: usize = if self == 0 { 1 } else { self.ilog(10) as usize + 1 };

                    // `buf` must have minimum size to store the decimal string version.
                    // BUF_SIZE is the size of the buffer.
                    if BUF_SIZE < decimal_string_size {
                        panic!("Not enough buffer size to format into!");
                    }

                    // Count the number of bytes in `buf` that are not initialized.
                    let mut offset = BUF_SIZE;

                    offset = raw_write_digits_into!(Self, BUF_SIZE)(self, buf, offset);

                    raw_extract_as_str(buf, offset)
                }
            }
        )*
    };
}

macro_impl_format_into! { unsigned; u8 u16 u32 u64 u128 usize }
macro_impl_format_into! { signed; (i8, u8) (i16, u16) (i32, u32) (i64, u64) (i128, u128) (isize, usize) }
