use clippy_utils::ty::{EnumValue, read_explicit_enum_value};
use rustc_middle::ty::{self, AdtDef, IntTy, Ty, TyCtxt, UintTy, VariantDiscr};

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

pub(super) fn enum_value_nbits(value: EnumValue) -> u64 {
    match value {
        EnumValue::Unsigned(x) => 128 - x.leading_zeros(),
        EnumValue::Signed(x) if x < 0 => 128 - (-(x + 1)).leading_zeros() + 1,
        EnumValue::Signed(x) => 128 - x.leading_zeros(),
    }
    .into()
}

pub(super) fn enum_ty_to_nbits(adt: AdtDef<'_>, tcx: TyCtxt<'_>) -> u64 {
    let mut explicit = 0i128;
    let (start, end) = adt
        .variants()
        .iter()
        .fold((0, i128::MIN), |(start, end), variant| match variant.discr {
            VariantDiscr::Relative(x) => match explicit.checked_add(i128::from(x)) {
                Some(x) => (start, end.max(x)),
                None => (i128::MIN, end),
            },
            VariantDiscr::Explicit(id) => match read_explicit_enum_value(tcx, id) {
                Some(EnumValue::Signed(x)) => {
                    explicit = x;
                    (start.min(x), end.max(x))
                },
                Some(EnumValue::Unsigned(x)) => match i128::try_from(x) {
                    Ok(x) => {
                        explicit = x;
                        (start, end.max(x))
                    },
                    Err(_) => (i128::MIN, end),
                },
                None => (start, end),
            },
        });

    if start > end {
        // No variants.
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
