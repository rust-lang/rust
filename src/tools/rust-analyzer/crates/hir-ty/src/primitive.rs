//! A few helper functions for dealing with primitives.

pub use chalk_ir::{FloatTy, IntTy, UintTy};
pub use hir_def::builtin_type::{BuiltinFloat, BuiltinInt, BuiltinUint};

pub fn int_ty_to_string(ty: IntTy) -> &'static str {
    match ty {
        IntTy::Isize => "isize",
        IntTy::I8 => "i8",
        IntTy::I16 => "i16",
        IntTy::I32 => "i32",
        IntTy::I64 => "i64",
        IntTy::I128 => "i128",
    }
}

pub fn uint_ty_to_string(ty: UintTy) -> &'static str {
    match ty {
        UintTy::Usize => "usize",
        UintTy::U8 => "u8",
        UintTy::U16 => "u16",
        UintTy::U32 => "u32",
        UintTy::U64 => "u64",
        UintTy::U128 => "u128",
    }
}

pub fn float_ty_to_string(ty: FloatTy) -> &'static str {
    match ty {
        FloatTy::F16 => "f16",
        FloatTy::F32 => "f32",
        FloatTy::F64 => "f64",
        FloatTy::F128 => "f128",
    }
}

pub(super) fn int_ty_from_builtin(t: BuiltinInt) -> IntTy {
    match t {
        BuiltinInt::Isize => IntTy::Isize,
        BuiltinInt::I8 => IntTy::I8,
        BuiltinInt::I16 => IntTy::I16,
        BuiltinInt::I32 => IntTy::I32,
        BuiltinInt::I64 => IntTy::I64,
        BuiltinInt::I128 => IntTy::I128,
    }
}

pub(super) fn uint_ty_from_builtin(t: BuiltinUint) -> UintTy {
    match t {
        BuiltinUint::Usize => UintTy::Usize,
        BuiltinUint::U8 => UintTy::U8,
        BuiltinUint::U16 => UintTy::U16,
        BuiltinUint::U32 => UintTy::U32,
        BuiltinUint::U64 => UintTy::U64,
        BuiltinUint::U128 => UintTy::U128,
    }
}

pub(super) fn float_ty_from_builtin(t: BuiltinFloat) -> FloatTy {
    match t {
        BuiltinFloat::F16 => FloatTy::F16,
        BuiltinFloat::F32 => FloatTy::F32,
        BuiltinFloat::F64 => FloatTy::F64,
        BuiltinFloat::F128 => FloatTy::F128,
    }
}
