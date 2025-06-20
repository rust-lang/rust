use rustc_abi::Size;
use rustc_ast::{self as ast};
use rustc_hir::LangItem;
use rustc_middle::bug;
use rustc_middle::mir::interpret::LitToConstInput;
use rustc_middle::ty::{self, ScalarInt, TyCtxt, TypeVisitableExt as _};
use tracing::trace;

use crate::builder::parse_float_into_scalar;

pub(crate) fn lit_to_const<'tcx>(
    tcx: TyCtxt<'tcx>,
    lit_input: LitToConstInput<'tcx>,
) -> ty::Const<'tcx> {
    let LitToConstInput { lit, ty, neg } = lit_input;

    if let Err(guar) = ty.error_reported() {
        return ty::Const::new_error(tcx, guar);
    }

    let trunc = |n, width: ty::UintTy| {
        let width = width
            .normalize(tcx.data_layout.pointer_size().bits().try_into().unwrap())
            .bit_width()
            .unwrap();
        let width = Size::from_bits(width);
        trace!("trunc {} with size {} and shift {}", n, width.bits(), 128 - width.bits());
        let result = width.truncate(n);
        trace!("trunc result: {}", result);

        ScalarInt::try_from_uint(result, width)
            .unwrap_or_else(|| bug!("expected to create ScalarInt from uint {:?}", result))
    };

    let valtree = match (lit, ty.kind()) {
        (ast::LitKind::Str(s, _), ty::Ref(_, inner_ty, _)) if inner_ty.is_str() => {
            let str_bytes = s.as_str().as_bytes();
            ty::ValTree::from_raw_bytes(tcx, str_bytes)
        }
        (ast::LitKind::Str(s, _), ty::Str) if tcx.features().deref_patterns() => {
            // String literal patterns may have type `str` if `deref_patterns` is enabled, in order
            // to allow `deref!("..."): String`.
            let str_bytes = s.as_str().as_bytes();
            ty::ValTree::from_raw_bytes(tcx, str_bytes)
        }
        (ast::LitKind::ByteStr(byte_sym, _), ty::Ref(_, inner_ty, _))
            if matches!(inner_ty.kind(), ty::Slice(_) | ty::Array(..)) =>
        {
            ty::ValTree::from_raw_bytes(tcx, byte_sym.as_byte_str())
        }
        (ast::LitKind::ByteStr(byte_sym, _), ty::Slice(_) | ty::Array(..))
            if tcx.features().deref_patterns() =>
        {
            // Byte string literal patterns may have type `[u8]` or `[u8; N]` if `deref_patterns` is
            // enabled, in order to allow, e.g., `deref!(b"..."): Vec<u8>`.
            ty::ValTree::from_raw_bytes(tcx, byte_sym.as_byte_str())
        }
        (ast::LitKind::Byte(n), ty::Uint(ty::UintTy::U8)) => {
            ty::ValTree::from_scalar_int(tcx, n.into())
        }
        (ast::LitKind::CStr(byte_sym, _), ty::Ref(_, inner_ty, _)) if matches!(inner_ty.kind(), ty::Adt(def, _) if tcx.is_lang_item(def.did(), LangItem::CStr)) => {
            ty::ValTree::from_raw_bytes(tcx, byte_sym.as_byte_str())
        }
        (ast::LitKind::Int(n, _), ty::Uint(ui)) if !neg => {
            let scalar_int = trunc(n.get(), *ui);
            ty::ValTree::from_scalar_int(tcx, scalar_int)
        }
        (ast::LitKind::Int(n, _), ty::Int(i)) => {
            let scalar_int = trunc(
                if neg { (n.get() as i128).overflowing_neg().0 as u128 } else { n.get() },
                i.to_unsigned(),
            );
            ty::ValTree::from_scalar_int(tcx, scalar_int)
        }
        (ast::LitKind::Bool(b), ty::Bool) => ty::ValTree::from_scalar_int(tcx, b.into()),
        (ast::LitKind::Float(n, _), ty::Float(fty)) => {
            let bits = parse_float_into_scalar(n, *fty, neg).unwrap_or_else(|| {
                tcx.dcx().bug(format!("couldn't parse float literal: {:?}", lit_input.lit))
            });
            ty::ValTree::from_scalar_int(tcx, bits)
        }
        (ast::LitKind::Char(c), ty::Char) => ty::ValTree::from_scalar_int(tcx, c.into()),
        (ast::LitKind::Err(guar), _) => return ty::Const::new_error(tcx, guar),
        _ => return ty::Const::new_misc_error(tcx),
    };

    ty::Const::new_value(tcx, valtree, ty)
}
