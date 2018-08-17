use rustc::ty::layout::{Size, HasDataLayout};
use rustc::mir::interpret::sign_extend;

use super::{Scalar, ScalarMaybeUndef, EvalResult};

pub trait ScalarExt {
    fn null(size: Size) -> Self;
    fn from_i32(i: i32) -> Self;
    fn from_uint(i: impl Into<u128>, ptr_size: Size) -> Self;
    fn from_int(i: impl Into<i128>, ptr_size: Size) -> Self;
    fn from_f32(f: f32) -> Self;
    fn from_f64(f: f64) -> Self;
    fn is_null(self) -> bool;
}

pub trait FalibleScalarExt {
    fn to_usize(self, cx: impl HasDataLayout) -> EvalResult<'static, u64>;
    fn to_isize(self, cx: impl HasDataLayout) -> EvalResult<'static, i64>;
    fn to_i32(self) -> EvalResult<'static, i32>;
    fn to_u8(self) -> EvalResult<'static, u8>;

    /// HACK: this function just extracts all bits if `defined != 0`
    /// Mainly used for args of C-functions and we should totally correctly fetch the size
    /// of their arguments
    fn to_bytes(self) -> EvalResult<'static, u128>;
}

impl ScalarExt for Scalar {
    fn null(size: Size) -> Self {
        Scalar::Bits { bits: 0, size: size.bytes() as u8 }
    }

    fn from_i32(i: i32) -> Self {
        Scalar::Bits { bits: i as u32 as u128, size: 4 }
    }

    fn from_uint(i: impl Into<u128>, size: Size) -> Self {
        Scalar::Bits { bits: i.into(), size: size.bytes() as u8 }
    }

    fn from_int(i: impl Into<i128>, size: Size) -> Self {
        Scalar::Bits { bits: i.into() as u128, size: size.bytes() as u8 }
    }

    fn from_f32(f: f32) -> Self {
        Scalar::Bits { bits: f.to_bits() as u128, size: 4 }
    }

    fn from_f64(f: f64) -> Self {
        Scalar::Bits { bits: f.to_bits() as u128, size: 8 }
    }

    fn is_null(self) -> bool {
        match self {
            Scalar::Bits { bits, .. } => bits == 0,
            Scalar::Ptr(_) => false
        }
    }
}

impl FalibleScalarExt for Scalar {
    fn to_usize(self, cx: impl HasDataLayout) -> EvalResult<'static, u64> {
        let b = self.to_bits(cx.data_layout().pointer_size)?;
        assert_eq!(b as u64 as u128, b);
        Ok(b as u64)
    }

    fn to_u8(self) -> EvalResult<'static, u8> {
        let sz = Size::from_bits(8);
        let b = self.to_bits(sz)?;
        assert_eq!(b as u8 as u128, b);
        Ok(b as u8)
    }

    fn to_isize(self, cx: impl HasDataLayout) -> EvalResult<'static, i64> {
        let b = self.to_bits(cx.data_layout().pointer_size)?;
        let b = sign_extend(b, cx.data_layout().pointer_size) as i128;
        assert_eq!(b as i64 as i128, b);
        Ok(b as i64)
    }

    fn to_i32(self) -> EvalResult<'static, i32> {
        let sz = Size::from_bits(32);
        let b = self.to_bits(sz)?;
        let b = sign_extend(b, sz) as i128;
        assert_eq!(b as i32 as i128, b);
        Ok(b as i32)
    }

    fn to_bytes(self) -> EvalResult<'static, u128> {
        match self {
            Scalar::Bits { bits, size } => {
                assert_ne!(size, 0);
                Ok(bits)
            },
            Scalar::Ptr(_) => err!(ReadPointerAsBytes),
        }
    }
}

impl FalibleScalarExt for ScalarMaybeUndef {
    fn to_usize(self, cx: impl HasDataLayout) -> EvalResult<'static, u64> {
        self.not_undef()?.to_usize(cx)
    }

    fn to_u8(self) -> EvalResult<'static, u8> {
        self.not_undef()?.to_u8()
    }

    fn to_isize(self, cx: impl HasDataLayout) -> EvalResult<'static, i64> {
        self.not_undef()?.to_isize(cx)
    }

    fn to_i32(self) -> EvalResult<'static, i32> {
        self.not_undef()?.to_i32()
    }

    fn to_bytes(self) -> EvalResult<'static, u128> {
        self.not_undef()?.to_bytes()
    }
}
