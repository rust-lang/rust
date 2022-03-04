use super::TRANSMUTE_UNDEFINED_REPR;
use clippy_utils::diagnostics::span_lint_and_then;
use clippy_utils::ty::is_c_void;
use rustc_hir::Expr;
use rustc_lint::LateContext;
use rustc_middle::ty::subst::Subst;
use rustc_middle::ty::{self, Ty, TypeAndMut};
use rustc_span::Span;

#[allow(clippy::too_many_lines)]
pub(super) fn check<'tcx>(
    cx: &LateContext<'tcx>,
    e: &'tcx Expr<'_>,
    from_ty_orig: Ty<'tcx>,
    to_ty_orig: Ty<'tcx>,
) -> bool {
    let mut from_ty = cx.tcx.erase_regions(from_ty_orig);
    let mut to_ty = cx.tcx.erase_regions(to_ty_orig);

    while from_ty != to_ty {
        match reduce_refs(cx, e.span, from_ty, to_ty) {
            ReducedTys::FromFatPtr {
                unsized_ty,
                to_ty: to_sub_ty,
            } => match reduce_ty(cx, to_sub_ty) {
                ReducedTy::IntArray | ReducedTy::TypeErasure => break,
                ReducedTy::Ref(to_sub_ty) => {
                    from_ty = unsized_ty;
                    to_ty = to_sub_ty;
                    continue;
                },
                _ => {
                    span_lint_and_then(
                        cx,
                        TRANSMUTE_UNDEFINED_REPR,
                        e.span,
                        &format!("transmute from `{}` which has an undefined layout", from_ty_orig),
                        |diag| {
                            if from_ty_orig.peel_refs() != unsized_ty {
                                diag.note(&format!("the contained type `&{}` has an undefined layout", unsized_ty));
                            }
                        },
                    );
                    return true;
                },
            },
            ReducedTys::ToFatPtr {
                unsized_ty,
                from_ty: from_sub_ty,
            } => match reduce_ty(cx, from_sub_ty) {
                ReducedTy::IntArray | ReducedTy::TypeErasure => break,
                ReducedTy::Ref(from_sub_ty) => {
                    from_ty = from_sub_ty;
                    to_ty = unsized_ty;
                    continue;
                },
                _ => {
                    span_lint_and_then(
                        cx,
                        TRANSMUTE_UNDEFINED_REPR,
                        e.span,
                        &format!("transmute to `{}` which has an undefined layout", to_ty_orig),
                        |diag| {
                            if to_ty_orig.peel_refs() != unsized_ty {
                                diag.note(&format!("the contained type `&{}` has an undefined layout", unsized_ty));
                            }
                        },
                    );
                    return true;
                },
            },
            ReducedTys::ToPtr {
                from_ty: from_sub_ty,
                to_ty: to_sub_ty,
            } => match reduce_ty(cx, from_sub_ty) {
                ReducedTy::UnorderedFields(from_ty) => {
                    span_lint_and_then(
                        cx,
                        TRANSMUTE_UNDEFINED_REPR,
                        e.span,
                        &format!("transmute from `{}` which has an undefined layout", from_ty_orig),
                        |diag| {
                            if from_ty_orig.peel_refs() != from_ty {
                                diag.note(&format!("the contained type `{}` has an undefined layout", from_ty));
                            }
                        },
                    );
                    return true;
                },
                ReducedTy::Ref(from_sub_ty) => {
                    from_ty = from_sub_ty;
                    to_ty = to_sub_ty;
                    continue;
                },
                _ => break,
            },
            ReducedTys::FromPtr {
                from_ty: from_sub_ty,
                to_ty: to_sub_ty,
            } => match reduce_ty(cx, to_sub_ty) {
                ReducedTy::UnorderedFields(to_ty) => {
                    span_lint_and_then(
                        cx,
                        TRANSMUTE_UNDEFINED_REPR,
                        e.span,
                        &format!("transmute to `{}` which has an undefined layout", to_ty_orig),
                        |diag| {
                            if to_ty_orig.peel_refs() != to_ty {
                                diag.note(&format!("the contained type `{}` has an undefined layout", to_ty));
                            }
                        },
                    );
                    return true;
                },
                ReducedTy::Ref(to_sub_ty) => {
                    from_ty = from_sub_ty;
                    to_ty = to_sub_ty;
                    continue;
                },
                _ => break,
            },
            ReducedTys::Other {
                from_ty: from_sub_ty,
                to_ty: to_sub_ty,
            } => match (reduce_ty(cx, from_sub_ty), reduce_ty(cx, to_sub_ty)) {
                (ReducedTy::IntArray | ReducedTy::TypeErasure, _)
                | (_, ReducedTy::IntArray | ReducedTy::TypeErasure) => return false,
                (ReducedTy::UnorderedFields(from_ty), ReducedTy::UnorderedFields(to_ty)) if from_ty != to_ty => {
                    span_lint_and_then(
                        cx,
                        TRANSMUTE_UNDEFINED_REPR,
                        e.span,
                        &format!(
                            "transmute from `{}` to `{}`, both of which have an undefined layout",
                            from_ty_orig, to_ty_orig
                        ),
                        |diag| {
                            if_chain! {
                                if let (Some(from_def), Some(to_def)) = (from_ty.ty_adt_def(), to_ty.ty_adt_def());
                                if from_def == to_def;
                                then {
                                    diag.note(&format!(
                                        "two instances of the same generic type (`{}`) may have different layouts",
                                        cx.tcx.item_name(from_def.did())
                                    ));
                                } else {
                                    if from_ty_orig.peel_refs() != from_ty {
                                        diag.note(&format!("the contained type `{}` has an undefined layout", from_ty));
                                    }
                                    if to_ty_orig.peel_refs() != to_ty {
                                        diag.note(&format!("the contained type `{}` has an undefined layout", to_ty));
                                    }
                                }
                            }
                        },
                    );
                    return true;
                },
                (
                    ReducedTy::UnorderedFields(from_ty),
                    ReducedTy::Other(_) | ReducedTy::OrderedFields(_) | ReducedTy::Ref(_),
                ) => {
                    span_lint_and_then(
                        cx,
                        TRANSMUTE_UNDEFINED_REPR,
                        e.span,
                        &format!("transmute from `{}` which has an undefined layout", from_ty_orig),
                        |diag| {
                            if from_ty_orig.peel_refs() != from_ty {
                                diag.note(&format!("the contained type `{}` has an undefined layout", from_ty));
                            }
                        },
                    );
                    return true;
                },
                (
                    ReducedTy::Other(_) | ReducedTy::OrderedFields(_) | ReducedTy::Ref(_),
                    ReducedTy::UnorderedFields(to_ty),
                ) => {
                    span_lint_and_then(
                        cx,
                        TRANSMUTE_UNDEFINED_REPR,
                        e.span,
                        &format!("transmute into `{}` which has an undefined layout", to_ty_orig),
                        |diag| {
                            if to_ty_orig.peel_refs() != to_ty {
                                diag.note(&format!("the contained type `{}` has an undefined layout", to_ty));
                            }
                        },
                    );
                    return true;
                },
                (ReducedTy::Ref(from_sub_ty), ReducedTy::Ref(to_sub_ty)) => {
                    from_ty = from_sub_ty;
                    to_ty = to_sub_ty;
                    continue;
                },
                (
                    ReducedTy::OrderedFields(_) | ReducedTy::Ref(_) | ReducedTy::Other(_),
                    ReducedTy::OrderedFields(_) | ReducedTy::Ref(_) | ReducedTy::Other(_),
                )
                | (ReducedTy::UnorderedFields(_), ReducedTy::UnorderedFields(_)) => break,
            },
        }
    }

    false
}

enum ReducedTys<'tcx> {
    FromFatPtr { unsized_ty: Ty<'tcx>, to_ty: Ty<'tcx> },
    ToFatPtr { unsized_ty: Ty<'tcx>, from_ty: Ty<'tcx> },
    ToPtr { from_ty: Ty<'tcx>, to_ty: Ty<'tcx> },
    FromPtr { from_ty: Ty<'tcx>, to_ty: Ty<'tcx> },
    Other { from_ty: Ty<'tcx>, to_ty: Ty<'tcx> },
}

/// Remove references so long as both types are references.
fn reduce_refs<'tcx>(
    cx: &LateContext<'tcx>,
    span: Span,
    mut from_ty: Ty<'tcx>,
    mut to_ty: Ty<'tcx>,
) -> ReducedTys<'tcx> {
    loop {
        return match (from_ty.kind(), to_ty.kind()) {
            (
                &(ty::Ref(_, from_sub_ty, _) | ty::RawPtr(TypeAndMut { ty: from_sub_ty, .. })),
                &(ty::Ref(_, to_sub_ty, _) | ty::RawPtr(TypeAndMut { ty: to_sub_ty, .. })),
            ) => {
                from_ty = from_sub_ty;
                to_ty = to_sub_ty;
                continue;
            },
            (&(ty::Ref(_, unsized_ty, _) | ty::RawPtr(TypeAndMut { ty: unsized_ty, .. })), _)
                if !unsized_ty.is_sized(cx.tcx.at(span), cx.param_env) =>
            {
                ReducedTys::FromFatPtr { unsized_ty, to_ty }
            },
            (_, &(ty::Ref(_, unsized_ty, _) | ty::RawPtr(TypeAndMut { ty: unsized_ty, .. })))
                if !unsized_ty.is_sized(cx.tcx.at(span), cx.param_env) =>
            {
                ReducedTys::ToFatPtr { unsized_ty, from_ty }
            },
            (&(ty::Ref(_, from_ty, _) | ty::RawPtr(TypeAndMut { ty: from_ty, .. })), _) => {
                ReducedTys::FromPtr { from_ty, to_ty }
            },
            (_, &(ty::Ref(_, to_ty, _) | ty::RawPtr(TypeAndMut { ty: to_ty, .. }))) => {
                ReducedTys::ToPtr { from_ty, to_ty }
            },
            _ => ReducedTys::Other { from_ty, to_ty },
        };
    }
}

enum ReducedTy<'tcx> {
    /// The type can be used for type erasure.
    TypeErasure,
    /// The type is a struct containing either zero non-zero sized fields, or multiple non-zero
    /// sized fields with a defined order.
    OrderedFields(Ty<'tcx>),
    /// The type is a struct containing multiple non-zero sized fields with no defined order.
    UnorderedFields(Ty<'tcx>),
    /// The type is a reference to the contained type.
    Ref(Ty<'tcx>),
    /// The type is an array of a primitive integer type. These can be used as storage for a value
    /// of another type.
    IntArray,
    /// Any other type.
    Other(Ty<'tcx>),
}

/// Reduce structs containing a single non-zero sized field to it's contained type.
fn reduce_ty<'tcx>(cx: &LateContext<'tcx>, mut ty: Ty<'tcx>) -> ReducedTy<'tcx> {
    loop {
        ty = cx.tcx.try_normalize_erasing_regions(cx.param_env, ty).unwrap_or(ty);
        return match *ty.kind() {
            ty::Array(sub_ty, _) if matches!(sub_ty.kind(), ty::Int(_) | ty::Uint(_)) => ReducedTy::IntArray,
            ty::Array(sub_ty, _) | ty::Slice(sub_ty) => {
                ty = sub_ty;
                continue;
            },
            ty::Tuple(args) if args.is_empty() => ReducedTy::TypeErasure,
            ty::Tuple(args) => {
                let Some(sized_ty) = args.iter().find(|&ty| !is_zero_sized_ty(cx, ty)) else {
                    return ReducedTy::OrderedFields(ty);
                };
                if args.iter().all(|ty| is_zero_sized_ty(cx, ty)) {
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
                    .map(|f| cx.tcx.type_of(f.did).subst(cx.tcx, substs));
                let Some(sized_ty) = iter.find(|&ty| !is_zero_sized_ty(cx, ty)) else {
                    return ReducedTy::TypeErasure;
                };
                if iter.all(|ty| is_zero_sized_ty(cx, ty)) {
                    ty = sized_ty;
                    continue;
                }
                if def.repr().inhibit_struct_field_reordering_opt() {
                    ReducedTy::OrderedFields(ty)
                } else {
                    ReducedTy::UnorderedFields(ty)
                }
            },
            ty::Adt(def, _) if def.is_enum() && (def.variants().is_empty() || is_c_void(cx, ty)) => {
                ReducedTy::TypeErasure
            },
            ty::Foreign(_) => ReducedTy::TypeErasure,
            ty::Ref(_, ty, _) => ReducedTy::Ref(ty),
            ty::RawPtr(ty) => ReducedTy::Ref(ty.ty),
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
