use clippy_utils::ty::{EnumValue, read_explicit_enum_value};
use rustc_middle::ty::{self, AdtDef, IntTy, Ty, TyCtxt, UintTy, VariantDiscr};

/// Returns the size in bits of an integral type, or `None` if `ty` is not an
/// integral type.
pub(super) fn int_ty_to_nbits(tcx: TyCtxt<'_>, ty: Ty<'_>) -> Option<u64> {
    match ty.kind() {
        ty::Int(IntTy::Isize) | ty::Uint(UintTy::Usize) => Some(tcx.data_layout.pointer_size().bits()),
        ty::Int(i) => i.bit_width(),
        ty::Uint(i) => i.bit_width(),
        _ => None,
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

pub(super) enum CastTo {
    Signed,
    Unsigned,
}
/// Returns `Some` if the type cast is between 2 integral types that differ
/// only in signedness, otherwise `None`. The value of `Some` is which
/// signedness is casted to.
pub(super) fn is_signedness_cast(cast_from: Ty<'_>, cast_to: Ty<'_>) -> Option<CastTo> {
    match (cast_from.kind(), cast_to.kind()) {
        (ty::Int(from), ty::Uint(to)) if from.to_unsigned() == *to => Some(CastTo::Unsigned),
        (ty::Uint(from), ty::Int(to)) if *from == to.to_unsigned() => Some(CastTo::Signed),
        _ => None,
    }
}
