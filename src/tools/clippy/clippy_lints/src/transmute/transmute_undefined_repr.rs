use super::TRANSMUTE_UNDEFINED_REPR;
use clippy_utils::diagnostics::span_lint_and_then;
use clippy_utils::ty::is_c_void;
use rustc_hir::Expr;
use rustc_lint::LateContext;
use rustc_middle::ty::SubstsRef;
use rustc_middle::ty::{self, IntTy, Ty, TypeAndMut, UintTy};

#[expect(clippy::too_many_lines)]
pub(super) fn check<'tcx>(
    cx: &LateContext<'tcx>,
    e: &'tcx Expr<'_>,
    from_ty_orig: Ty<'tcx>,
    to_ty_orig: Ty<'tcx>,
) -> bool {
    let mut from_ty = cx.tcx.erase_regions(from_ty_orig);
    let mut to_ty = cx.tcx.erase_regions(to_ty_orig);

    while from_ty != to_ty {
        let reduced_tys = reduce_refs(cx, from_ty, to_ty);
        match (reduce_ty(cx, reduced_tys.from_ty), reduce_ty(cx, reduced_tys.to_ty)) {
            // Various forms of type erasure.
            (ReducedTy::TypeErasure { raw_ptr_only: false }, _)
            | (_, ReducedTy::TypeErasure { raw_ptr_only: false }) => return false,
            (ReducedTy::TypeErasure { .. }, _) if reduced_tys.from_raw_ptr => return false,
            (_, ReducedTy::TypeErasure { .. }) if reduced_tys.to_raw_ptr => return false,

            // `Repr(C)` <-> unordered type.
            // If the first field of the `Repr(C)` type matches then the transmute is ok
            (ReducedTy::OrderedFields(_, Some(from_sub_ty)), ReducedTy::UnorderedFields(to_sub_ty))
            | (ReducedTy::UnorderedFields(from_sub_ty), ReducedTy::OrderedFields(_, Some(to_sub_ty))) => {
                from_ty = from_sub_ty;
                to_ty = to_sub_ty;
                continue;
            },
            (ReducedTy::OrderedFields(_, Some(from_sub_ty)), ReducedTy::Other(to_sub_ty)) if reduced_tys.to_fat_ptr => {
                from_ty = from_sub_ty;
                to_ty = to_sub_ty;
                continue;
            },
            (ReducedTy::Other(from_sub_ty), ReducedTy::OrderedFields(_, Some(to_sub_ty)))
                if reduced_tys.from_fat_ptr =>
            {
                from_ty = from_sub_ty;
                to_ty = to_sub_ty;
                continue;
            },

            // ptr <-> ptr
            (ReducedTy::Other(from_sub_ty), ReducedTy::Other(to_sub_ty))
                if matches!(from_sub_ty.kind(), ty::Ref(..) | ty::RawPtr(_))
                    && matches!(to_sub_ty.kind(), ty::Ref(..) | ty::RawPtr(_)) =>
            {
                from_ty = from_sub_ty;
                to_ty = to_sub_ty;
                continue;
            },

            // fat ptr <-> (*size, *size)
            (ReducedTy::Other(_), ReducedTy::UnorderedFields(to_ty))
                if reduced_tys.from_fat_ptr && is_size_pair(to_ty) =>
            {
                return false;
            },
            (ReducedTy::UnorderedFields(from_ty), ReducedTy::Other(_))
                if reduced_tys.to_fat_ptr && is_size_pair(from_ty) =>
            {
                return false;
            },

            // fat ptr -> some struct | some struct -> fat ptr
            (ReducedTy::Other(_), _) if reduced_tys.from_fat_ptr => {
                span_lint_and_then(
                    cx,
                    TRANSMUTE_UNDEFINED_REPR,
                    e.span,
                    &format!("transmute from `{from_ty_orig}` which has an undefined layout"),
                    |diag| {
                        if from_ty_orig.peel_refs() != from_ty.peel_refs() {
                            diag.note(&format!("the contained type `{from_ty}` has an undefined layout"));
                        }
                    },
                );
                return true;
            },
            (_, ReducedTy::Other(_)) if reduced_tys.to_fat_ptr => {
                span_lint_and_then(
                    cx,
                    TRANSMUTE_UNDEFINED_REPR,
                    e.span,
                    &format!("transmute to `{to_ty_orig}` which has an undefined layout"),
                    |diag| {
                        if to_ty_orig.peel_refs() != to_ty.peel_refs() {
                            diag.note(&format!("the contained type `{to_ty}` has an undefined layout"));
                        }
                    },
                );
                return true;
            },

            (ReducedTy::UnorderedFields(from_ty), ReducedTy::UnorderedFields(to_ty)) if from_ty != to_ty => {
                let same_adt_did = if let (ty::Adt(from_def, from_subs), ty::Adt(to_def, to_subs))
                        = (from_ty.kind(), to_ty.kind())
                        && from_def == to_def
                    {
                        if same_except_params(from_subs, to_subs) {
                            return false;
                        }
                        Some(from_def.did())
                    } else {
                        None
                    };
                span_lint_and_then(
                    cx,
                    TRANSMUTE_UNDEFINED_REPR,
                    e.span,
                    &format!(
                        "transmute from `{from_ty_orig}` to `{to_ty_orig}`, both of which have an undefined layout"
                    ),
                    |diag| {
                        if let Some(same_adt_did) = same_adt_did {
                            diag.note(&format!(
                                "two instances of the same generic type (`{}`) may have different layouts",
                                cx.tcx.item_name(same_adt_did)
                            ));
                        } else {
                            if from_ty_orig.peel_refs() != from_ty {
                                diag.note(&format!("the contained type `{from_ty}` has an undefined layout"));
                            }
                            if to_ty_orig.peel_refs() != to_ty {
                                diag.note(&format!("the contained type `{to_ty}` has an undefined layout"));
                            }
                        }
                    },
                );
                return true;
            },
            (
                ReducedTy::UnorderedFields(from_ty),
                ReducedTy::Other(_) | ReducedTy::OrderedFields(..) | ReducedTy::TypeErasure { raw_ptr_only: true },
            ) => {
                span_lint_and_then(
                    cx,
                    TRANSMUTE_UNDEFINED_REPR,
                    e.span,
                    &format!("transmute from `{from_ty_orig}` which has an undefined layout"),
                    |diag| {
                        if from_ty_orig.peel_refs() != from_ty {
                            diag.note(&format!("the contained type `{from_ty}` has an undefined layout"));
                        }
                    },
                );
                return true;
            },
            (
                ReducedTy::Other(_) | ReducedTy::OrderedFields(..) | ReducedTy::TypeErasure { raw_ptr_only: true },
                ReducedTy::UnorderedFields(to_ty),
            ) => {
                span_lint_and_then(
                    cx,
                    TRANSMUTE_UNDEFINED_REPR,
                    e.span,
                    &format!("transmute into `{to_ty_orig}` which has an undefined layout"),
                    |diag| {
                        if to_ty_orig.peel_refs() != to_ty {
                            diag.note(&format!("the contained type `{to_ty}` has an undefined layout"));
                        }
                    },
                );
                return true;
            },
            (
                ReducedTy::OrderedFields(..) | ReducedTy::Other(_) | ReducedTy::TypeErasure { raw_ptr_only: true },
                ReducedTy::OrderedFields(..) | ReducedTy::Other(_) | ReducedTy::TypeErasure { raw_ptr_only: true },
            )
            | (ReducedTy::UnorderedFields(_), ReducedTy::UnorderedFields(_)) => {
                break;
            },
        }
    }

    false
}

#[expect(clippy::struct_excessive_bools)]
struct ReducedTys<'tcx> {
    from_ty: Ty<'tcx>,
    to_ty: Ty<'tcx>,
    from_raw_ptr: bool,
    to_raw_ptr: bool,
    from_fat_ptr: bool,
    to_fat_ptr: bool,
}

/// Remove references so long as both types are references.
fn reduce_refs<'tcx>(cx: &LateContext<'tcx>, mut from_ty: Ty<'tcx>, mut to_ty: Ty<'tcx>) -> ReducedTys<'tcx> {
    let mut from_raw_ptr = false;
    let mut to_raw_ptr = false;
    let (from_fat_ptr, to_fat_ptr) = loop {
        break match (from_ty.kind(), to_ty.kind()) {
            (
                &(ty::Ref(_, from_sub_ty, _) | ty::RawPtr(TypeAndMut { ty: from_sub_ty, .. })),
                &(ty::Ref(_, to_sub_ty, _) | ty::RawPtr(TypeAndMut { ty: to_sub_ty, .. })),
            ) => {
                from_raw_ptr = matches!(*from_ty.kind(), ty::RawPtr(_));
                from_ty = from_sub_ty;
                to_raw_ptr = matches!(*to_ty.kind(), ty::RawPtr(_));
                to_ty = to_sub_ty;
                continue;
            },
            (&(ty::Ref(_, unsized_ty, _) | ty::RawPtr(TypeAndMut { ty: unsized_ty, .. })), _)
                if !unsized_ty.is_sized(cx.tcx, cx.param_env) =>
            {
                (true, false)
            },
            (_, &(ty::Ref(_, unsized_ty, _) | ty::RawPtr(TypeAndMut { ty: unsized_ty, .. })))
                if !unsized_ty.is_sized(cx.tcx, cx.param_env) =>
            {
                (false, true)
            },
            _ => (false, false),
        };
    };
    ReducedTys {
        from_ty,
        to_ty,
        from_raw_ptr,
        to_raw_ptr,
        from_fat_ptr,
        to_fat_ptr,
    }
}

enum ReducedTy<'tcx> {
    /// The type can be used for type erasure.
    TypeErasure { raw_ptr_only: bool },
    /// The type is a struct containing either zero non-zero sized fields, or multiple non-zero
    /// sized fields with a defined order.
    /// The second value is the first non-zero sized type.
    OrderedFields(Ty<'tcx>, Option<Ty<'tcx>>),
    /// The type is a struct containing multiple non-zero sized fields with no defined order.
    UnorderedFields(Ty<'tcx>),
    /// Any other type.
    Other(Ty<'tcx>),
}

/// Reduce structs containing a single non-zero sized field to it's contained type.
fn reduce_ty<'tcx>(cx: &LateContext<'tcx>, mut ty: Ty<'tcx>) -> ReducedTy<'tcx> {
    loop {
        ty = cx.tcx.try_normalize_erasing_regions(cx.param_env, ty).unwrap_or(ty);
        return match *ty.kind() {
            ty::Array(sub_ty, _) if matches!(sub_ty.kind(), ty::Int(_) | ty::Uint(_)) => {
                ReducedTy::TypeErasure { raw_ptr_only: false }
            },
            ty::Array(sub_ty, _) | ty::Slice(sub_ty) => {
                ty = sub_ty;
                continue;
            },
            ty::Tuple(args) if args.is_empty() => ReducedTy::TypeErasure { raw_ptr_only: false },
            ty::Tuple(args) => {
                let mut iter = args.iter();
                let Some(sized_ty) = iter.find(|&ty| !is_zero_sized_ty(cx, ty)) else {
                    return ReducedTy::OrderedFields(ty, None);
                };
                if iter.all(|ty| is_zero_sized_ty(cx, ty)) {
                    ty = sized_ty;
                    continue;
                }
                ReducedTy::UnorderedFields(ty)
            },
            ty::Adt(def, substs) if def.is_struct() => {
                let mut iter = def
                    .non_enum_variant()
                    .fields
                    .iter()
                    .map(|f| cx.tcx.bound_type_of(f.did).subst(cx.tcx, substs));
                let Some(sized_ty) = iter.find(|&ty| !is_zero_sized_ty(cx, ty)) else {
                    return ReducedTy::TypeErasure { raw_ptr_only: false };
                };
                if iter.all(|ty| is_zero_sized_ty(cx, ty)) {
                    ty = sized_ty;
                    continue;
                }
                if def.repr().inhibit_struct_field_reordering_opt() {
                    ReducedTy::OrderedFields(ty, Some(sized_ty))
                } else {
                    ReducedTy::UnorderedFields(ty)
                }
            },
            ty::Adt(def, _) if def.is_enum() && (def.variants().is_empty() || is_c_void(cx, ty)) => {
                ReducedTy::TypeErasure { raw_ptr_only: false }
            },
            // TODO: Check if the conversion to or from at least one of a union's fields is valid.
            ty::Adt(def, _) if def.is_union() => ReducedTy::TypeErasure { raw_ptr_only: false },
            ty::Foreign(_) | ty::Param(_) => ReducedTy::TypeErasure { raw_ptr_only: false },
            ty::Int(_) | ty::Uint(_) => ReducedTy::TypeErasure { raw_ptr_only: true },
            _ => ReducedTy::Other(ty),
        };
    }
}

fn is_zero_sized_ty<'tcx>(cx: &LateContext<'tcx>, ty: Ty<'tcx>) -> bool {
    if_chain! {
        if let Ok(ty) = cx.tcx.try_normalize_erasing_regions(cx.param_env, ty);
        if let Ok(layout) = cx.tcx.layout_of(cx.param_env.and(ty));
        then {
            layout.layout.size().bytes() == 0
        } else {
            false
        }
    }
}

fn is_size_pair(ty: Ty<'_>) -> bool {
    if let ty::Tuple(tys) = *ty.kind()
        && let [ty1, ty2] = &**tys
    {
        matches!(ty1.kind(), ty::Int(IntTy::Isize) | ty::Uint(UintTy::Usize))
            && matches!(ty2.kind(), ty::Int(IntTy::Isize) | ty::Uint(UintTy::Usize))
    } else {
        false
    }
}

fn same_except_params<'tcx>(subs1: SubstsRef<'tcx>, subs2: SubstsRef<'tcx>) -> bool {
    // TODO: check const parameters as well. Currently this will consider `Array<5>` the same as
    // `Array<6>`
    for (ty1, ty2) in subs1.types().zip(subs2.types()).filter(|(ty1, ty2)| ty1 != ty2) {
        match (ty1.kind(), ty2.kind()) {
            (ty::Param(_), _) | (_, ty::Param(_)) => (),
            (ty::Adt(adt1, subs1), ty::Adt(adt2, subs2)) if adt1 == adt2 && same_except_params(subs1, subs2) => (),
            _ => return false,
        }
    }
    true
}
