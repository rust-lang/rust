//! This module check that we are not transmuting `&T` to `&mut T` or `&UnsafeCell<T>`.
//!
//! The complexity comes from the fact that we want to lint against this cast even when the `&T`, the `&mut T`
//! or the `&UnsafeCell<T>` (either the `&` or the `UnsafeCell` part of it) are hidden in fields.
//!
//! The general idea is to first isolate potential candidates, then check if they are really problematic.
//!
//! That is, we first collect all instances of `&mut` references in the target type, then we check if any
//! of them overlap with a `&` reference in the source type.
//!
//! We do a similar thing to `UnsafeCell`. Here it is a little more complicated, since we operate two layers deep:
//! first, we collect all `&` references in the target type. Then, we check if any of them overlaps with a `&` reference
//! in the source type (not a `&mut` reference, since it is perfectly fine to transmute `&mut T` to `&UnsafeCell<T>`).
//! If we found such overlap, we collect all instances of `UnsafeCell` under the reference in the target type.
//! If we found any, we try to find if it overlaps with things that are not `UnsafeCell` under the reference
//! in the source type.

use std::fmt::Write;

use rustc_hir::def::{DefKind, Res};
use rustc_hir::def_id::DefId;
use rustc_hir::{BorrowKind, Expr, ExprKind, UnOp};
use rustc_index::IndexSlice;
use rustc_middle::bug;
use rustc_middle::ty::layout::LayoutOf;
use rustc_middle::ty::{self, AdtDef, GenericArgsRef, List, Mutability, Ty};
use rustc_session::{declare_lint, declare_lint_pass};
use rustc_span::symbol::sym;
use rustc_target::abi::{FieldIdx, FieldsShape, HasDataLayout, Layout, Size};
use tracing::{debug, debug_span};

use crate::lints::{BuiltinMutablesTransmutes, TransmuteBreadcrumbs, UnsafeCellTransmutes};
use crate::{LateContext, LateLintPass, LintContext};

fn struct_fields<'a, 'tcx>(
    cx: &'a LateContext<'tcx>,
    layout: Layout<'tcx>,
    adt_def: AdtDef<'tcx>,
    substs: GenericArgsRef<'tcx>,
) -> impl Iterator<Item = (FieldIdx, Size, Ty<'tcx>)> + 'a {
    let field_tys = adt_def
        .non_enum_variant()
        .fields
        .iter()
        .map(|field| cx.tcx.normalize_erasing_regions(cx.param_env, field.ty(cx.tcx, substs)));
    // I thought this could panic, but apparently SIMD vectors are structs but also have primitive layout. So...
    let field_offsets = match &layout.fields {
        FieldsShape::Arbitrary { offsets, .. } => offsets.as_slice(),
        _ => IndexSlice::empty(),
    };
    std::iter::zip(field_offsets.iter_enumerated(), field_tys)
        .map(|((idx, &offset), ty)| (idx, offset, ty))
}

fn tuple_fields<'tcx>(
    layout: Layout<'tcx>,
    field_tys: &'tcx List<Ty<'tcx>>,
) -> impl Iterator<Item = (FieldIdx, Size, Ty<'tcx>)> + 'tcx {
    let field_offsets = match &layout.fields {
        FieldsShape::Arbitrary { offsets, .. } => offsets,
        _ => bug!("expected FieldsShape::Arbitrary for tuples"),
    };
    std::iter::zip(field_offsets.iter_enumerated(), field_tys)
        .map(|((idx, &offset), ty)| (idx, offset, ty))
}

#[derive(Debug, Default, Clone)]
struct FieldsBreadcrumbs(Vec<FieldIdx>);

impl FieldsBreadcrumbs {
    fn push_with<R>(&mut self, field: FieldIdx, f: impl FnOnce(&mut FieldsBreadcrumbs) -> R) -> R {
        self.0.push(field);
        let result = f(self);
        self.0.pop();
        result
    }

    fn translate_for_diagnostics<'tcx>(
        &self,
        cx: &LateContext<'tcx>,
        output: &mut String,
        mut ty: Ty<'tcx>,
    ) {
        for &current_breadcrumb in &self.0 {
            ty = match ty.kind() {
                &ty::Adt(adt_def, substs) if adt_def.is_struct() => {
                    let field = &adt_def.non_enum_variant().fields[current_breadcrumb];
                    let field_ty =
                        cx.tcx.normalize_erasing_regions(cx.param_env, field.ty(cx.tcx, substs));
                    output.push_str(".");
                    output.push_str(field.name.as_str());

                    field_ty
                }

                &ty::Tuple(tys) => {
                    let field_ty = tys[current_breadcrumb.as_usize()];
                    write!(output, ".{}", current_breadcrumb.as_u32()).unwrap();

                    field_ty
                }

                &ty::Array(element_ty, _) => {
                    output.push_str("[0]");

                    element_ty
                }

                _ => bug!("field breadcrumb on an unknown type"),
            }
        }
    }

    // Formats the breadcrumbs as `(*path.to.reference).path.to.unsafecell`;
    fn format_unsafe_cell_for_diag<'tcx>(
        cx: &LateContext<'tcx>,
        reference: &FieldsBreadcrumbs,
        top_ty: Ty<'tcx>,
        referent: &FieldsBreadcrumbs,
        reference_ty: Ty<'tcx>,
    ) -> TransmuteBreadcrumbs<'tcx> {
        let mut after_ty = String::new();
        let before_ty;
        if !referent.0.is_empty() {
            before_ty = "(*";
            reference.translate_for_diagnostics(cx, &mut after_ty, top_ty);
            after_ty.push_str(")");
            referent.translate_for_diagnostics(cx, &mut after_ty, reference_ty);
        } else {
            before_ty = "*";
            reference.translate_for_diagnostics(cx, &mut after_ty, top_ty);
        }
        TransmuteBreadcrumbs { before_ty, ty: top_ty, after_ty }
    }

    // Formats the breadcrumbs as `(*path.to.reference).path.to.unsafecell`;
    fn format_for_diag<'tcx>(
        &self,
        cx: &LateContext<'tcx>,
        ty: Ty<'tcx>,
    ) -> TransmuteBreadcrumbs<'tcx> {
        let mut after_ty = String::new();
        if !self.0.is_empty() {
            self.translate_for_diagnostics(cx, &mut after_ty, ty);
        }
        TransmuteBreadcrumbs { before_ty: "", ty, after_ty }
    }
}

#[derive(PartialEq)]
enum CollectFields {
    Yes,
    No,
}

fn collect_fields<'tcx>(
    cx: &LateContext<'tcx>,
    ty: Ty<'tcx>,
    offset: Size,
    callback: &mut impl FnMut(Size, Ty<'tcx>, &FieldsBreadcrumbs) -> CollectFields,
    fields_breadcrumbs: &mut FieldsBreadcrumbs,
) {
    if callback(offset, ty, &fields_breadcrumbs) == CollectFields::No {
        return;
    }

    match ty.kind() {
        &ty::Adt(adt_def, substs) if adt_def.is_struct() => {
            let Ok(layout) = cx.layout_of(ty) else {
                return;
            };
            debug!(?offset, ?layout, "collect_fields struct");
            for (field_idx, field_offset, field_ty) in
                struct_fields(cx, layout.layout, adt_def, substs)
            {
                fields_breadcrumbs.push_with(field_idx, |fields_breadcrumbs| {
                    collect_fields(
                        cx,
                        field_ty,
                        offset + field_offset,
                        callback,
                        fields_breadcrumbs,
                    )
                });
            }
        }

        &ty::Tuple(tys) => {
            let Ok(layout) = cx.layout_of(ty) else {
                return;
            };
            debug!(?offset, ?layout, "collect_fields tuple");
            for (field_idx, field_offset, field_ty) in tuple_fields(layout.layout, tys) {
                fields_breadcrumbs.push_with(field_idx, |fields_breadcrumbs| {
                    collect_fields(
                        cx,
                        field_ty,
                        offset + field_offset,
                        callback,
                        fields_breadcrumbs,
                    )
                });
            }
        }

        &ty::Array(ty, len)
            if let Some(len) = len.try_eval_target_usize(cx.tcx, cx.param_env)
                && len != 0 =>
        {
            debug!(?offset, "collect_fields array");
            fields_breadcrumbs.push_with(FieldIdx::from_u32(0), |fields_breadcrumbs| {
                collect_fields(cx, ty, offset, callback, fields_breadcrumbs)
            });
        }

        _ => {}
    }
}

trait Field {
    fn start_offset(&self) -> Size;
    fn end_offset(&self) -> Size;
}

#[derive(Debug)]
enum CheckOverlapping {
    Yes,
    No,
}

/// `check_overlap_with` must be sorted by starting offset.
fn check_overlapping<'tcx>(
    cx: &LateContext<'tcx>,
    check_overlap_with: &[impl Field],
    ty: Ty<'tcx>,
    offset: Size,
    on_overlapping: &mut impl FnMut(Ty<'tcx>, usize, &FieldsBreadcrumbs) -> CheckOverlapping,
    fields_breadcrumbs: &mut FieldsBreadcrumbs,
) {
    let Ok(layout) = cx.layout_of(ty) else {
        return;
    };
    debug!(?offset, ?layout, "check_overlapping");

    // Then, for any entry that we overlap with (there can be many as our size can be bigger than one,
    // or because both are only partially overlapping) call the closure.
    let mut idx = 0;
    while idx < check_overlap_with.len()
        && check_overlap_with[idx].start_offset() < offset + layout.size
    {
        if check_overlap_with[idx].end_offset() <= offset {
            idx += 1;
            continue;
        }

        let span = debug_span!("on_overlapping", ?ty, ?idx, ?fields_breadcrumbs);
        let _guard = span.enter();
        let result = on_overlapping(ty, idx, &fields_breadcrumbs);
        debug!(?result, "on_overlapping result");
        match result {
            CheckOverlapping::Yes => {}
            CheckOverlapping::No => return,
        };
        idx += 1;
    }

    // Finally, descend to every field and check there too.
    match ty.kind() {
        &ty::Adt(adt_def, substs) if adt_def.is_struct() => {
            for (field_idx, field_offset, field_ty) in
                struct_fields(cx, layout.layout, adt_def, substs)
            {
                fields_breadcrumbs.push_with(field_idx, |fields_breadcrumbs| {
                    check_overlapping(
                        cx,
                        check_overlap_with,
                        field_ty,
                        offset + field_offset,
                        on_overlapping,
                        fields_breadcrumbs,
                    )
                });
            }
        }

        &ty::Tuple(tys) => {
            for (field_idx, field_offset, field_ty) in tuple_fields(layout.layout, tys) {
                fields_breadcrumbs.push_with(field_idx, |fields_breadcrumbs| {
                    check_overlapping(
                        cx,
                        check_overlap_with,
                        field_ty,
                        offset + field_offset,
                        on_overlapping,
                        fields_breadcrumbs,
                    )
                });
            }
        }

        &ty::Array(ty, _) if layout.size != Size::ZERO => {
            fields_breadcrumbs.push_with(FieldIdx::from_u32(0), |fields_breadcrumbs| {
                check_overlapping(
                    cx,
                    check_overlap_with,
                    ty,
                    offset,
                    on_overlapping,
                    fields_breadcrumbs,
                )
            });
        }

        _ => {}
    }
}

#[derive(Debug)]
struct Ref<'tcx> {
    start_offset: Size,
    end_offset: Size,
    ty: Ty<'tcx>,
    mutability: Mutability,
    fields_breadcrumbs: FieldsBreadcrumbs,
}

impl Field for Ref<'_> {
    fn start_offset(&self) -> Size {
        self.start_offset
    }

    fn end_offset(&self) -> Size {
        self.end_offset
    }
}

#[derive(Debug)]
struct UnsafeCellField {
    start_offset: Size,
    end_offset: Size,
    fields_breadcrumbs: FieldsBreadcrumbs,
}

impl Field for UnsafeCellField {
    fn start_offset(&self) -> Size {
        self.start_offset
    }

    fn end_offset(&self) -> Size {
        self.end_offset
    }
}

/// The parameters to `report_error` are: breadcrumbs to src's UnsafeCell, breadcrumbs to dst's UnsafeCell.
fn check_unsafe_cells<'tcx>(
    cx: &LateContext<'tcx>,
    dst_ref_ty: Ty<'tcx>,
    src_ty: Ty<'tcx>,
    mut report_error: impl FnMut(&FieldsBreadcrumbs, &FieldsBreadcrumbs),
) {
    let mut dst_unsafe_cells = Vec::new();
    collect_fields(
        cx,
        dst_ref_ty,
        Size::ZERO,
        &mut |offset, ty, fields_breadcrumbs| match ty.kind() {
            &ty::Adt(adt_def, _) if adt_def.is_unsafe_cell() => {
                let Ok(layout) = cx.layout_of(ty) else {
                    return CollectFields::Yes;
                };
                if layout.is_zst() {
                    return CollectFields::No;
                }

                let Ok(layout) = cx.layout_of(ty) else {
                    return CollectFields::Yes;
                };
                dst_unsafe_cells.push(UnsafeCellField {
                    start_offset: offset,
                    end_offset: offset + layout.size,
                    fields_breadcrumbs: fields_breadcrumbs.clone(),
                });
                CollectFields::No
            }
            _ => CollectFields::Yes,
        },
        &mut FieldsBreadcrumbs::default(),
    );
    dst_unsafe_cells.sort_unstable_by_key(|unsafe_cell| unsafe_cell.start_offset);
    debug!(?dst_unsafe_cells, "collected UnsafeCells");

    check_overlapping(
        cx,
        &dst_unsafe_cells,
        src_ty,
        Size::ZERO,
        &mut |ty, idx, src_pointee_fields_breadcrumbs| {
            // First, check that there are actually some bytes there, not just padding or ZST.
            match ty.kind() {
                // Transmuting `&UnsafeCell` to `&UnsafeCell` is fine, don't check inside.
                ty::Adt(adt_def, _) if adt_def.is_unsafe_cell() => {
                    return CheckOverlapping::No;
                }

                ty::Bool
                | ty::Char
                | ty::Int(..)
                | ty::Uint(..)
                | ty::Float(..)
                | ty::RawPtr(..)
                | ty::Ref(..)
                | ty::FnPtr(..) => {}
                // Enums have their discriminant.
                ty::Adt(adt_def, _) if adt_def.is_enum() => {}
                _ => return CheckOverlapping::Yes,
            }

            report_error(src_pointee_fields_breadcrumbs, &dst_unsafe_cells[idx].fields_breadcrumbs);

            CheckOverlapping::No
        },
        &mut FieldsBreadcrumbs::default(),
    );
}

/// Checks for transmutes via `std::mem::transmute` and lints.
fn lint_transmutes(cx: &LateContext<'_>, expr: &Expr<'_>) {
    let Some((src, dst)) = get_transmute_from_to(cx, expr) else { return };
    let span = debug_span!("lint_transmutes");
    let _guard = span.enter();
    debug!(?src, ?dst);

    let mut dst_refs = Vec::new();
    // First, collect references in the target type, so we can see for mutable references if they are transmuted
    // from a shared reference, and for shared references if `UnsafeCell` inside them is transmuted from non-`UnsafeCell`.
    // Instead of doing this in two passes, one for shared references (`UnsafeCell`) and another for mutable,
    // we do it in one pass and keep track of what kind of reference it is.
    collect_fields(
        cx,
        dst,
        Size::ZERO,
        &mut |offset, ty, fields_breadcrumbs| match ty.kind() {
            &ty::Ref(_, ty, mutability) => {
                dst_refs.push(Ref {
                    start_offset: offset,
                    end_offset: offset + cx.data_layout().pointer_size,
                    ty,
                    mutability,
                    fields_breadcrumbs: fields_breadcrumbs.clone(),
                });
                CollectFields::No
            }
            _ => CollectFields::Yes,
        },
        &mut FieldsBreadcrumbs::default(),
    );
    dst_refs.sort_unstable_by_key(|ref_| ref_.start_offset);
    debug!(?dst_refs, "collected refs");
    check_overlapping(
        cx,
        &dst_refs,
        src,
        Size::ZERO,
        &mut |ty, idx, src_fields_breadcrumbs| {
            let dst_ref = &dst_refs[idx];
            match dst_ref.mutability {
                // For mutable references in the target type, we need to see if they were converted from shared references.
                Mutability::Mut => match ty.kind() {
                    ty::Ref(_, _, Mutability::Not) => {
                        let from = src_fields_breadcrumbs.format_for_diag(cx, src);
                        let to = dst_ref.fields_breadcrumbs.format_for_diag(cx, dst);
                        cx.emit_span_lint(
                            MUTABLE_TRANSMUTES,
                            expr.span,
                            BuiltinMutablesTransmutes { from, to },
                        );
                        CheckOverlapping::No
                    }
                    _ => CheckOverlapping::Yes,
                },
                // For shared references, we need to see if they were transmuted from shared (not mutable!) references,
                // and if there is an incorrectly transmuted `UnsafeCell` inside.
                Mutability::Not => {
                    let &ty::Ref(_, src_ty, Mutability::Not) = ty.kind() else {
                        return CheckOverlapping::Yes;
                    };
                    debug!(?src_ty, dst_ty = ?dst_ref.ty, "found shared ref");

                    check_unsafe_cells(
                        cx,
                        dst_ref.ty,
                        src_ty,
                        |src_pointee_fields_breadcrumbs, dst_pointee_fields_breadcrumbs| {
                            let from = FieldsBreadcrumbs::format_unsafe_cell_for_diag(
                                cx,
                                src_fields_breadcrumbs,
                                src,
                                src_pointee_fields_breadcrumbs,
                                src_ty,
                            );
                            let to = FieldsBreadcrumbs::format_unsafe_cell_for_diag(
                                cx,
                                &dst_ref.fields_breadcrumbs,
                                dst,
                                dst_pointee_fields_breadcrumbs,
                                dst_ref.ty,
                            );

                            cx.emit_span_lint(
                                UNSAFE_CELL_TRANSMUTES,
                                expr.span,
                                UnsafeCellTransmutes { from, to, orig_cast: None },
                            );
                        },
                    );
                    CheckOverlapping::No
                }
            }
        },
        &mut FieldsBreadcrumbs::default(),
    );
}

fn get_transmute_from_to<'tcx>(
    cx: &LateContext<'tcx>,
    expr: &Expr<'_>,
) -> Option<(Ty<'tcx>, Ty<'tcx>)> {
    let def = if let ExprKind::Path(ref qpath) = expr.kind {
        cx.qpath_res(qpath, expr.hir_id)
    } else {
        return None;
    };
    if let Res::Def(DefKind::Fn, did) = def {
        if !def_id_is_transmute(cx, did) {
            return None;
        }
        let sig = cx.typeck_results().node_type(expr.hir_id).fn_sig(cx.tcx);
        let from = sig.inputs().skip_binder()[0];
        let to = sig.output().skip_binder();
        return Some((from, to));
    }
    None
}

fn def_id_is_transmute(cx: &LateContext<'_>, def_id: DefId) -> bool {
    cx.tcx.is_intrinsic(def_id, sym::transmute)
}

/// Checks for transmutes via `&*(reference as *const _ as *const _)` and lints.
fn lint_reference_casting<'tcx>(cx: &LateContext<'tcx>, expr: &'tcx Expr<'tcx>) {
    let Some(e) = borrow_or_assign(expr) else {
        return;
    };

    let init = cx.expr_or_init(e);

    let Some((dst, src)) = is_type_cast(cx, init) else {
        return;
    };
    let orig_cast = if init.span != e.span { Some(init.span) } else { None };

    let span = debug_span!("lint_reference_casting");
    let _guard = span.enter();

    check_unsafe_cells(cx, dst, src, |src_fields_breadcrumbs, dst_fields_breadcrumbs| {
        let from = src_fields_breadcrumbs.format_for_diag(cx, src);
        let to = dst_fields_breadcrumbs.format_for_diag(cx, dst);
        cx.emit_span_lint(UNSAFE_CELL_TRANSMUTES, expr.span, UnsafeCellTransmutes {
            from,
            to,
            orig_cast,
        });
    });
}

fn borrow_or_assign<'tcx>(expr: &'tcx Expr<'tcx>) -> Option<&'tcx Expr<'tcx>> {
    // &(mut) <expr>
    let inner = if let ExprKind::AddrOf(BorrowKind::Ref, Mutability::Not, expr) = expr.kind {
        expr
    // <expr> = ...
    } else if let ExprKind::Assign(expr, _, _) = expr.kind {
        expr
    // <expr> += ...
    } else if let ExprKind::AssignOp(_, expr, _) = expr.kind {
        expr
    } else {
        return None;
    };

    // *<inner>
    let ExprKind::Unary(UnOp::Deref, e) = &inner.kind else {
        return None;
    };
    Some(e)
}

fn is_type_cast<'tcx>(
    cx: &LateContext<'tcx>,
    orig_expr: &'tcx Expr<'tcx>,
) -> Option<(Ty<'tcx>, Ty<'tcx>)> {
    let mut e = orig_expr;

    let end_ty = cx.typeck_results().node_type(orig_expr.hir_id);

    let &ty::RawPtr(dst, _) = end_ty.kind() else {
        return None;
    };

    loop {
        e = e.peel_blocks();
        // <expr> as ...
        e = if let ExprKind::Cast(expr, _) = e.kind {
            expr
        // <expr>.cast()
        } else if let ExprKind::MethodCall(_, expr, [], _) = e.kind
            && let Some(def_id) = cx.typeck_results().type_dependent_def_id(e.hir_id)
            && matches!(
                cx.tcx.get_diagnostic_name(def_id),
                Some(sym::ptr_cast | sym::const_ptr_cast),
            )
        {
            expr
        // ptr::from_ref(<expr>) or mem::transmute<_, _>(<expr>)
        } else if let ExprKind::Call(path, [arg]) = e.kind
            && let ExprKind::Path(ref qpath) = path.kind
            && let Some(def_id) = cx.qpath_res(qpath, path.hir_id).opt_def_id()
            && matches!(
                cx.tcx.get_diagnostic_name(def_id),
                Some(sym::ptr_from_ref | sym::transmute)
            )
        {
            arg
        } else {
            break;
        };
    }

    let &ty::Ref(_, src, Mutability::Not) = cx.typeck_results().node_type(e.hir_id).kind() else {
        return None;
    };

    Some((dst, src))
}

declare_lint! {
    /// The `mutable_transmutes` lint catches transmuting from `&T` to `&mut
    /// T` because it is [undefined behavior].
    ///
    /// [undefined behavior]: https://doc.rust-lang.org/reference/behavior-considered-undefined.html
    ///
    /// ### Example
    ///
    /// ```rust,compile_fail
    /// unsafe {
    ///     let y = std::mem::transmute::<&i32, &mut i32>(&5);
    /// }
    /// ```
    ///
    /// {{produces}}
    ///
    /// ### Explanation
    ///
    /// Certain assumptions are made about aliasing of data, and this transmute
    /// violates those assumptions. Consider using [`UnsafeCell`] instead.
    ///
    /// [`UnsafeCell`]: https://doc.rust-lang.org/std/cell/struct.UnsafeCell.html
    pub MUTABLE_TRANSMUTES,
    Deny,
    "transmuting &T to &mut T is undefined behavior, even if the reference is unused"
}

declare_lint! {
    /// The `unsafe_cell_transmutes` lint catches transmuting or casting from `&T` to [`&UnsafeCell<T>`]
    /// because it dangerous and might be [undefined behavior].
    ///
    /// [undefined behavior]: https://doc.rust-lang.org/reference/behavior-considered-undefined.html
    ///
    /// ### Example
    ///
    /// ```rust,compile_fail
    /// use std::cell::Cell;
    ///
    /// unsafe {
    ///     let x = 5_i32;
    ///     let y = std::mem::transmute::<&i32, &Cell<i32>>(&x);
    ///     y.set(6);
    ///
    ///     let z = &*(&x as *const i32 as *const Cell<i32>);
    ///     z.set(7);
    /// }
    /// ```
    ///
    /// {{produces}}
    ///
    /// ### Explanation
    ///
    /// Conversion from `&T` to `&UnsafeCell<T>` might be immediate undefined behavior, depending on
    /// unspecified details of the aliasing model.
    ///
    /// Even if it is not, writing to it will be undefined behavior if there was no `UnsafeCell` in
    /// the original `T`, and even if there was, it might be undefined behavior (again, depending
    /// on unspecified details of the aliasing model).
    ///
    /// It is also highly dangerous and error-prone, and unlikely to be useful.
    ///
    /// [`&UnsafeCell<T>`]: https://doc.rust-lang.org/std/cell/struct.UnsafeCell.html
    pub UNSAFE_CELL_TRANSMUTES,
    Deny,
    "transmuting &T to &UnsafeCell<T> is error-prone, rarely intentional and may cause undefined behavior"
}

declare_lint_pass!(MutableTransmutes => [MUTABLE_TRANSMUTES, UNSAFE_CELL_TRANSMUTES]);

impl<'tcx> LateLintPass<'tcx> for MutableTransmutes {
    fn check_expr(&mut self, cx: &LateContext<'tcx>, expr: &'tcx Expr<'tcx>) {
        lint_reference_casting(cx, expr);
        lint_transmutes(cx, expr);
    }
}
