use rustc_middle::mir::interpret::{ConstValue, Scalar};
use rustc_middle::ty::{self, AdtDef, IntTy, Ty, TyCtxt, UintTy, VariantDiscr};
use rustc_target::abi::Size;

/// Returns the size in bits of an integral type.
/// Will return 0 if the type is not an int or uint variant
pub(super) fn int_ty_to_nbits(typ: Ty<'_>, tcx: TyCtxt<'_>) -> u64 {
    match typ.kind() {
        ty::Int(i) => match i {
            IntTy::Isize => tcx.data_layout.pointer_size.bits(),
            IntTy::I8 => 8,
            IntTy::I16 => 16,
            IntTy::I32 => 32,
            IntTy::I64 => 64,
            IntTy::I128 => 128,
        },
        ty::Uint(i) => match i {
            UintTy::Usize => tcx.data_layout.pointer_size.bits(),
            UintTy::U8 => 8,
            UintTy::U16 => 16,
            UintTy::U32 => 32,
            UintTy::U64 => 64,
            UintTy::U128 => 128,
        },
        _ => 0,
    }
}

pub(super) fn enum_ty_to_nbits(adt: &AdtDef, tcx: TyCtxt<'_>) -> u64 {
    let mut explicit = 0i128;
    let (start, end) = adt
        .variants
        .iter()
        .fold((i128::MAX, i128::MIN), |(start, end), variant| match variant.discr {
            VariantDiscr::Relative(x) => match explicit.checked_add(i128::from(x)) {
                Some(x) => (start, end.max(x)),
                None => (i128::MIN, end),
            },
            VariantDiscr::Explicit(id) => {
                let ty = tcx.type_of(id);
                if let Ok(ConstValue::Scalar(Scalar::Int(value))) = tcx.const_eval_poly(id) {
                    #[allow(clippy::cast_possible_truncation, clippy::cast_possible_wrap)]
                    let value = match (value.size().bytes(), ty.kind()) {
                        (1, ty::Int(_)) => i128::from(value.assert_bits(Size::from_bytes(1)) as u8 as i8),
                        (1, ty::Uint(_)) => i128::from(value.assert_bits(Size::from_bytes(1)) as u8),
                        (2, ty::Int(_)) => i128::from(value.assert_bits(Size::from_bytes(2)) as u16 as i16),
                        (2, ty::Uint(_)) => i128::from(value.assert_bits(Size::from_bytes(2)) as u16),
                        (4, ty::Int(_)) => i128::from(value.assert_bits(Size::from_bytes(4)) as u32 as i32),
                        (4, ty::Uint(_)) => i128::from(value.assert_bits(Size::from_bytes(4)) as u32),
                        (8, ty::Int(_)) => i128::from(value.assert_bits(Size::from_bytes(8)) as u64 as i64),
                        (8, ty::Uint(_)) => i128::from(value.assert_bits(Size::from_bytes(8)) as u64),
                        (16, ty::Int(_)) => value.assert_bits(Size::from_bytes(16)) as i128,
                        (16, ty::Uint(_)) => match i128::try_from(value.assert_bits(Size::from_bytes(16))) {
                            Ok(x) => x,
                            // Requires 128 bits
                            Err(_) => return (i128::MIN, end),
                        },
                        // Shouldn't happen if compilation was successful
                        _ => return (start, end),
                    };
                    explicit = value;
                    (start.min(value), end.max(value))
                } else {
                    // Shouldn't happen if compilation was successful
                    (start, end)
                }
            },
        });

    if start >= end {
        0
    } else {
        let neg_bits = if start < 0 {
            128 - (-(start + 1)).leading_zeros() + 1
        } else {
            0
        };
        let pos_bits = if end > 0 { 128 - end.leading_zeros() } else { 0 };
        neg_bits.max(pos_bits).into()
    }
}
