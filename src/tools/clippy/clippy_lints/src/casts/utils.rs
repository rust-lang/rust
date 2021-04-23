use rustc_middle::ty::{self, IntTy, Ty, TyCtxt, UintTy};

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
