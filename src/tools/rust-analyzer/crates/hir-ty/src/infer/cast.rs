//! Type cast logic. Basically coercion + additional casts.

use crate::{infer::unify::InferenceTable, Interner, Ty, TyExt, TyKind};

#[derive(Clone, Debug)]
pub(super) struct CastCheck {
    expr_ty: Ty,
    cast_ty: Ty,
}

impl CastCheck {
    pub(super) fn new(expr_ty: Ty, cast_ty: Ty) -> Self {
        Self { expr_ty, cast_ty }
    }

    pub(super) fn check(self, table: &mut InferenceTable<'_>) {
        // FIXME: This function currently only implements the bits that influence the type
        // inference. We should return the adjustments on success and report diagnostics on error.
        let expr_ty = table.resolve_ty_shallow(&self.expr_ty);
        let cast_ty = table.resolve_ty_shallow(&self.cast_ty);

        if expr_ty.contains_unknown() || cast_ty.contains_unknown() {
            return;
        }

        if table.coerce(&expr_ty, &cast_ty).is_ok() {
            return;
        }

        if check_ref_to_ptr_cast(expr_ty, cast_ty, table) {
            // Note that this type of cast is actually split into a coercion to a
            // pointer type and a cast:
            // &[T; N] -> *[T; N] -> *T
            return;
        }

        // FIXME: Check other kinds of non-coercion casts and report error if any?
    }
}

fn check_ref_to_ptr_cast(expr_ty: Ty, cast_ty: Ty, table: &mut InferenceTable<'_>) -> bool {
    let Some((expr_inner_ty, _, _)) = expr_ty.as_reference() else {
        return false;
    };
    let Some((cast_inner_ty, _)) = cast_ty.as_raw_ptr() else {
        return false;
    };
    let TyKind::Array(expr_elt_ty, _) = expr_inner_ty.kind(Interner) else {
        return false;
    };
    table.coerce(expr_elt_ty, cast_inner_ty).is_ok()
}
