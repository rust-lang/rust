#![macro_use]

macro_rules! max_leb128_len {
    ($int_ty:ty) => {
        // The longest LEB128 encoding for an integer uses 7 bits per byte.
        (std::mem::size_of::<$int_ty>() * 8 + 6) / 7
    };
}

// Returns the longest LEB128 encoding of all supported integer types.
pub const fn max_leb128_len() -> usize {
    max_leb128_len!(u128)
}

macro_rules! impl_write_unsigned_leb128 {
    ($fn_name:ident, $int_ty:ty) => {
        #[inline]
        pub fn $fn_name(
            out: &mut [::std::mem::MaybeUninit<u8>; max_leb128_len!($int_ty)],
            mut value: $int_ty,
        ) -> &[u8] {
            let mut i = 0;

            loop {
                if value < 0x80 {
                    unsafe {
                        *out.get_unchecked_mut(i).as_mut_ptr() = value as u8;
                    }

                    i += 1;
                    break;
                } else {
                    unsafe {
                        *out.get_unchecked_mut(i).as_mut_ptr() = ((value & 0x7f) | 0x80) as u8;
                    }

                    value >>= 7;
                    i += 1;
                }
            }

            unsafe { ::std::mem::MaybeUninit::slice_assume_init_ref(&out.get_unchecked(..i)) }
        }
    };
}

impl_write_unsigned_leb128!(write_u16_leb128, u16);
impl_write_unsigned_leb128!(write_u32_leb128, u32);
impl_write_unsigned_leb128!(write_u64_leb128, u64);
impl_write_unsigned_leb128!(write_u128_leb128, u128);
impl_write_unsigned_leb128!(write_usize_leb128, usize);

macro_rules! impl_read_unsigned_leb128 {
    ($fn_name:ident, $int_ty:ty) => {
        #[inline]
        pub fn $fn_name(slice: &[u8], position: &mut usize) -> $int_ty {
            // The first iteration of this loop is unpeeled. This is a
            // performance win because this code is hot and integer values less
            // than 128 are very common, typically occurring 50-80% or more of
            // the time, even for u64 and u128.
            let byte = slice[*position];
            *position += 1;
            if (byte & 0x80) == 0 {
                return byte as $int_ty;
            }
            let mut result = (byte & 0x7F) as $int_ty;
            let mut shift = 7;
            loop {
                let byte = slice[*position];
                *position += 1;
                if (byte & 0x80) == 0 {
                    result |= (byte as $int_ty) << shift;
                    return result;
                } else {
                    result |= ((byte & 0x7F) as $int_ty) << shift;
                }
                shift += 7;
            }
        }
    };
}

impl_read_unsigned_leb128!(read_u16_leb128, u16);
impl_read_unsigned_leb128!(read_u32_leb128, u32);
impl_read_unsigned_leb128!(read_u64_leb128, u64);
impl_read_unsigned_leb128!(read_u128_leb128, u128);
impl_read_unsigned_leb128!(read_usize_leb128, usize);

macro_rules! impl_write_signed_leb128 {
    ($fn_name:ident, $int_ty:ty) => {
        #[inline]
        pub fn $fn_name(
            out: &mut [::std::mem::MaybeUninit<u8>; max_leb128_len!($int_ty)],
            mut value: $int_ty,
        ) -> &[u8] {
            let mut i = 0;

            loop {
                let mut byte = (value as u8) & 0x7f;
                value >>= 7;
                let more = !(((value == 0) && ((byte & 0x40) == 0))
                    || ((value == -1) && ((byte & 0x40) != 0)));

                if more {
                    byte |= 0x80; // Mark this byte to show that more bytes will follow.
                }

                unsafe {
                    *out.get_unchecked_mut(i).as_mut_ptr() = byte;
                }

                i += 1;

                if !more {
                    break;
                }
            }

            unsafe { ::std::mem::MaybeUninit::slice_assume_init_ref(&out.get_unchecked(..i)) }
        }
    };
}

impl_write_signed_leb128!(write_i16_leb128, i16);
impl_write_signed_leb128!(write_i32_leb128, i32);
impl_write_signed_leb128!(write_i64_leb128, i64);
impl_write_signed_leb128!(write_i128_leb128, i128);
impl_write_signed_leb128!(write_isize_leb128, isize);

macro_rules! impl_read_signed_leb128 {
    ($fn_name:ident, $int_ty:ty) => {
        #[inline]
        pub fn $fn_name(slice: &[u8], position: &mut usize) -> $int_ty {
            let mut result = 0;
            let mut shift = 0;
            let mut byte;

            loop {
                byte = slice[*position];
                *position += 1;
                result |= <$int_ty>::from(byte & 0x7F) << shift;
                shift += 7;

                if (byte & 0x80) == 0 {
                    break;
                }
            }

            if (shift < <$int_ty>::BITS) && ((byte & 0x40) != 0) {
                // sign extend
                result |= (!0 << shift);
            }

            result
        }
    };
}

impl_read_signed_leb128!(read_i16_leb128, i16);
impl_read_signed_leb128!(read_i32_leb128, i32);
impl_read_signed_leb128!(read_i64_leb128, i64);
impl_read_signed_leb128!(read_i128_leb128, i128);
impl_read_signed_leb128!(read_isize_leb128, isize);
