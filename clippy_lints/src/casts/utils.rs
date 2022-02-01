use rustc_middle::mir::interpret::{ConstValue, Scalar};
use rustc_middle::ty::{self, AdtDef, IntTy, Ty, TyCtxt, UintTy, VariantDiscr};
use rustc_span::def_id::DefId;
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

pub(super) enum EnumValue {
    Unsigned(u128),
    Signed(i128),
}
impl EnumValue {
    pub(super) fn add(self, n: u32) -> Self {
        match self {
            Self::Unsigned(x) => Self::Unsigned(x + u128::from(n)),
            Self::Signed(x) => Self::Signed(x + i128::from(n)),
        }
    }

    pub(super) fn nbits(self) -> u64 {
        match self {
            Self::Unsigned(x) => 128 - x.leading_zeros(),
            Self::Signed(x) if x < 0 => 128 - (-(x + 1)).leading_zeros() + 1,
            Self::Signed(x) => 128 - x.leading_zeros(),
        }
        .into()
    }
}

#[allow(clippy::cast_possible_truncation, clippy::cast_possible_wrap)]
pub(super) fn read_explicit_enum_value(tcx: TyCtxt<'_>, id: DefId) -> Option<EnumValue> {
    if let Ok(ConstValue::Scalar(Scalar::Int(value))) = tcx.const_eval_poly(id) {
        match tcx.type_of(id).kind() {
            ty::Int(_) => Some(EnumValue::Signed(match value.size().bytes() {
                1 => i128::from(value.assert_bits(Size::from_bytes(1)) as u8 as i8),
                2 => i128::from(value.assert_bits(Size::from_bytes(2)) as u16 as i16),
                4 => i128::from(value.assert_bits(Size::from_bytes(4)) as u32 as i32),
                8 => i128::from(value.assert_bits(Size::from_bytes(8)) as u64 as i64),
                16 => value.assert_bits(Size::from_bytes(16)) as i128,
                _ => return None,
            })),
            ty::Uint(_) => Some(EnumValue::Unsigned(match value.size().bytes() {
                1 => value.assert_bits(Size::from_bytes(1)),
                2 => value.assert_bits(Size::from_bytes(2)),
                4 => value.assert_bits(Size::from_bytes(4)),
                8 => value.assert_bits(Size::from_bytes(8)),
                16 => value.assert_bits(Size::from_bytes(16)),
                _ => return None,
            })),
            _ => None,
        }
    } else {
        None
    }
}

pub(super) fn enum_ty_to_nbits(adt: &AdtDef, tcx: TyCtxt<'_>) -> u64 {
    let mut explicit = 0i128;
    let (start, end) = adt
        .variants
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
