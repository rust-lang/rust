use rustc_abi::Size;
use rustc_ast::{self as ast, UintTy};
use rustc_hir::LangItem;
use rustc_middle::bug;
use rustc_middle::ty::{self, LitToConstInput, ScalarInt, Ty, TyCtxt, TypeVisitableExt as _};
use tracing::trace;

use crate::builder::parse_float_into_scalar;

pub(crate) fn lit_to_const<'tcx>(
    tcx: TyCtxt<'tcx>,
    lit_input: LitToConstInput<'tcx>,
) -> Option<ty::Value<'tcx>> {
    let LitToConstInput { lit, ty: expected_ty, neg } = lit_input;

    if expected_ty.error_reported().is_err() {
        return None;
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

    let (valtree, valtree_ty) = match (lit, expected_ty.kind()) {
        (ast::LitKind::Str(s, _), _) => {
            let str_bytes = s.as_str().as_bytes();
            let valtree_ty = Ty::new_imm_ref(tcx, tcx.lifetimes.re_static, tcx.types.str_);
            (ty::ValTree::from_raw_bytes(tcx, str_bytes), valtree_ty)
        }
        (ast::LitKind::ByteStr(byte_sym, _), ty::Ref(_, inner_ty, _))
            if let ty::Slice(ty) | ty::Array(ty, _) = inner_ty.kind()
                && let ty::Uint(UintTy::U8) = ty.kind() =>
        {
            (ty::ValTree::from_raw_bytes(tcx, byte_sym.as_byte_str()), expected_ty)
        }
        (ast::LitKind::ByteStr(byte_sym, _), ty::Slice(inner_ty) | ty::Array(inner_ty, _))
            if tcx.features().deref_patterns()
                && let ty::Uint(UintTy::U8) = inner_ty.kind() =>
        {
            // Byte string literal patterns may have type `[u8]` or `[u8; N]` if `deref_patterns` is
            // enabled, in order to allow, e.g., `deref!(b"..."): Vec<u8>`.
            (ty::ValTree::from_raw_bytes(tcx, byte_sym.as_byte_str()), expected_ty)
        }
        (ast::LitKind::ByteStr(byte_sym, _), _) => {
            let valtree = ty::ValTree::from_raw_bytes(tcx, byte_sym.as_byte_str());
            let valtree_ty = Ty::new_array(tcx, tcx.types.u8, byte_sym.as_byte_str().len() as u64);
            (valtree, valtree_ty)
        }
        (ast::LitKind::Byte(n), _) => (ty::ValTree::from_scalar_int(tcx, n.into()), tcx.types.u8),
        (ast::LitKind::CStr(byte_sym, _), _)
            if let Some(cstr_def_id) = tcx.lang_items().get(LangItem::CStr) =>
        {
            // A CStr is a newtype around a byte slice, so we create the inner slice here.
            // We need a branch for each "level" of the data structure.
            let cstr_ty = tcx.type_of(cstr_def_id).skip_binder();
            let bytes = ty::ValTree::from_raw_bytes(tcx, byte_sym.as_byte_str());
            let valtree =
                ty::ValTree::from_branches(tcx, [ty::Const::new_value(tcx, bytes, cstr_ty)]);
            let valtree_ty = Ty::new_imm_ref(tcx, tcx.lifetimes.re_static, cstr_ty);
            (valtree, valtree_ty)
        }
        (ast::LitKind::Int(n, ast::LitIntType::Unsigned(ui)), _) if !neg => {
            let scalar_int = trunc(n.get(), ui);
            (ty::ValTree::from_scalar_int(tcx, scalar_int), Ty::new_uint(tcx, ui))
        }
        (ast::LitKind::Int(_, ast::LitIntType::Unsigned(_)), _) if neg => return None,
        (ast::LitKind::Int(n, ast::LitIntType::Signed(i)), _) => {
            let scalar_int =
                trunc(if neg { u128::wrapping_neg(n.get()) } else { n.get() }, i.to_unsigned());
            (ty::ValTree::from_scalar_int(tcx, scalar_int), Ty::new_int(tcx, i))
        }
        (ast::LitKind::Int(n, ast::LitIntType::Unsuffixed), ty::Uint(ui)) if !neg => {
            let scalar_int = trunc(n.get(), *ui);
            (ty::ValTree::from_scalar_int(tcx, scalar_int), Ty::new_uint(tcx, *ui))
        }
        (ast::LitKind::Int(n, ast::LitIntType::Unsuffixed), ty::Int(i)) => {
            // Unsigned "negation" has the same bitwise effect as signed negation,
            // which gets the result we want without additional casts.
            let scalar_int =
                trunc(if neg { u128::wrapping_neg(n.get()) } else { n.get() }, i.to_unsigned());
            (ty::ValTree::from_scalar_int(tcx, scalar_int), Ty::new_int(tcx, *i))
        }
        (ast::LitKind::Bool(b), _) => (ty::ValTree::from_scalar_int(tcx, b.into()), tcx.types.bool),
        (ast::LitKind::Float(n, ast::LitFloatType::Suffixed(fty)), _) => {
            let fty = match fty {
                ast::FloatTy::F16 => ty::FloatTy::F16,
                ast::FloatTy::F32 => ty::FloatTy::F32,
                ast::FloatTy::F64 => ty::FloatTy::F64,
                ast::FloatTy::F128 => ty::FloatTy::F128,
            };
            let bits = parse_float_into_scalar(n, fty, neg)?;
            (ty::ValTree::from_scalar_int(tcx, bits), Ty::new_float(tcx, fty))
        }
        (ast::LitKind::Float(n, ast::LitFloatType::Unsuffixed), ty::Float(fty)) => {
            let bits = parse_float_into_scalar(n, *fty, neg)?;
            (ty::ValTree::from_scalar_int(tcx, bits), Ty::new_float(tcx, *fty))
        }
        (ast::LitKind::Char(c), _) => (ty::ValTree::from_scalar_int(tcx, c.into()), tcx.types.char),
        (ast::LitKind::Err(_), _) => return None,
        _ => return None,
    };

    Some(ty::Value { ty: valtree_ty, valtree })
}
