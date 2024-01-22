//! This module check that we are not transmuting `&T` to `&mut T` or `&UnsafeCell<T>`.
//!
//! The complexity comes from the fact that we want to lint against this cast even the `&T`, the `&mut T`
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

use crate::{
    lints::{BuiltinMutablesTransmutes, BuiltinUnsafeCellTransmutes},
    LateContext, LateLintPass, LintContext,
};
use rustc_hir as hir;
use rustc_hir::def::{DefKind, Res};
use rustc_hir::def_id::DefId;
use rustc_index::IndexSlice;
use rustc_middle::ty::layout::LayoutOf;
use rustc_middle::ty::{self, AdtDef, GenericArgsRef, List, Ty};
use rustc_span::symbol::sym;
use rustc_target::abi::{FieldIdx, FieldsShape, HasDataLayout, Layout, Size};
use rustc_type_ir::Mutability;

use std::fmt::Write;
use std::ops::ControlFlow;

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
    MUTABLE_TRANSMUTES,
    Deny,
    "transmuting &T to &mut T is undefined behavior, even if the reference is unused"
}

declare_lint! {
    /// The `unsafe_cell_transmutes` lint catches transmuting from `&T` to [`&UnsafeCell<T>`]
    /// because it is [undefined behavior].
    ///
    /// [undefined behavior]: https://doc.rust-lang.org/reference/behavior-considered-undefined.html
    ///
    /// ### Example
    ///
    /// ```rust,compile_fail
    /// use std::cell::UnsafeCell;
    ///
    /// unsafe {
    ///     let y = std::mem::transmute::<&i32, &UnsafeCell<i32>>(&5);
    /// }
    /// ```
    ///
    /// {{produces}}
    ///
    /// ### Explanation
    ///
    /// Certain assumptions are made about aliasing of data, and this transmute
    /// violates those assumptions. Consider using `UnsafeCell` for the original data.
    ///
    /// [`&UnsafeCell<T>`]: https://doc.rust-lang.org/std/cell/struct.UnsafeCell.html
    UNSAFE_CELL_TRANSMUTES,
    Deny,
    "transmuting &T to &UnsafeCell<T> is undefined behavior, even if the reference is unused"
}

declare_lint_pass!(MutableTransmutes => [MUTABLE_TRANSMUTES, UNSAFE_CELL_TRANSMUTES]);

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
        let mut breadcrumbs = self.0.as_slice();
        while let &[current_breadcrumb, ref rest_breadcrumbs @ ..] = breadcrumbs {
            breadcrumbs = rest_breadcrumbs;

            match ty.kind() {
                &ty::Adt(adt_def, substs) if adt_def.is_struct() => {
                    let field = &adt_def.non_enum_variant().fields[current_breadcrumb];
                    let field_ty =
                        cx.tcx.normalize_erasing_regions(cx.param_env, field.ty(cx.tcx, substs));
                    output.push_str(".");
                    output.push_str(field.name.as_str());

                    ty = field_ty;
                }

                &ty::Tuple(tys) => {
                    let field_ty = tys[current_breadcrumb.as_usize()];
                    write!(output, ".{}", current_breadcrumb.as_u32()).unwrap();

                    ty = field_ty;
                }

                &ty::Array(element_ty, _) => {
                    output.push_str("[0]");

                    ty = element_ty;
                }

                _ => bug!("field breadcrumb on an unknown type"),
            }
        }
    }

    // Formats the breadcrumbs as `(*path.to.reference).path.to.unsafecell`;
    fn unsafe_cell_breadcrumbs<'tcx>(
        cx: &LateContext<'tcx>,
        reference: &FieldsBreadcrumbs,
        top_ty: Ty<'tcx>,
        referent: &FieldsBreadcrumbs,
        reference_ty: Ty<'tcx>,
    ) -> String {
        let mut result;
        if !referent.0.is_empty() {
            result = format!("(*{top_ty}");
            reference.translate_for_diagnostics(cx, &mut result, top_ty);
            result.push_str(")");
            referent.translate_for_diagnostics(cx, &mut result, reference_ty);
        } else {
            result = format!("*{top_ty}");
            reference.translate_for_diagnostics(cx, &mut result, top_ty);
        }
        result
    }
}

#[derive(PartialEq)]
enum CollectFields {
    CheckFields,
    DoNotCheckFields,
}

fn collect_fields<'tcx>(
    cx: &LateContext<'tcx>,
    ty: Ty<'tcx>,
    offset: Size,
    callback: &mut impl FnMut(Size, Ty<'tcx>, &FieldsBreadcrumbs) -> CollectFields,
    fields_breadcrumbs: &mut FieldsBreadcrumbs,
) {
    if callback(offset, ty, &fields_breadcrumbs) == CollectFields::DoNotCheckFields {
        return;
    }

    match ty.kind() {
        &ty::Adt(adt_def, substs) if adt_def.is_struct() => {
            let Ok(layout) = cx.layout_of(ty) else {
                return;
            };
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

enum CheckOverlapping {
    CheckFields,
    DoNotCheckFields,
    /// Reported an error - stop the search completely (in order to not report more than one error).
    ReportedError,
}

/// `check_overlap_with` must be sorted by starting offset.
fn check_overlapping<'tcx>(
    cx: &LateContext<'tcx>,
    check_overlap_with: &[impl Field],
    mut first_maybe_overlapping: usize,
    ty: Ty<'tcx>,
    offset: Size,
    on_overlapping: &mut dyn FnMut(Ty<'tcx>, usize, &FieldsBreadcrumbs) -> CheckOverlapping,
    fields_breadcrumbs: &mut FieldsBreadcrumbs,
) -> ControlFlow<()> {
    let Ok(layout) = cx.layout_of(ty) else {
        return ControlFlow::Continue(());
    };

    // First, advance the cursor to the first potentially overlapping entry,
    // in order to speed up lookup for fields.
    while first_maybe_overlapping < check_overlap_with.len()
        && check_overlap_with[first_maybe_overlapping].end_offset() < offset
    {
        first_maybe_overlapping += 1;
    }

    if first_maybe_overlapping >= check_overlap_with.len() {
        return ControlFlow::Continue(());
    }

    // Then, for any entry that we overlap with (there can be many as our size can be bigger than one,
    // or because both are only partially overlapping) call the closure.
    let mut idx = first_maybe_overlapping;
    while idx < check_overlap_with.len()
        && check_overlap_with[idx].start_offset() < offset + layout.size
    {
        match on_overlapping(ty, idx, &fields_breadcrumbs) {
            CheckOverlapping::CheckFields => {}
            CheckOverlapping::DoNotCheckFields => return ControlFlow::Continue(()),
            CheckOverlapping::ReportedError => return ControlFlow::Break(()),
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
                        first_maybe_overlapping,
                        field_ty,
                        offset + field_offset,
                        on_overlapping,
                        fields_breadcrumbs,
                    )
                })?;
            }
        }

        &ty::Tuple(tys) => {
            for (field_idx, field_offset, field_ty) in tuple_fields(layout.layout, tys) {
                fields_breadcrumbs.push_with(field_idx, |fields_breadcrumbs| {
                    check_overlapping(
                        cx,
                        check_overlap_with,
                        first_maybe_overlapping,
                        field_ty,
                        offset + field_offset,
                        on_overlapping,
                        fields_breadcrumbs,
                    )
                })?;
            }
        }

        &ty::Array(ty, _) if layout.size != Size::ZERO => {
            fields_breadcrumbs.push_with(FieldIdx::from_u32(0), |fields_breadcrumbs| {
                check_overlapping(
                    cx,
                    check_overlap_with,
                    first_maybe_overlapping,
                    ty,
                    offset,
                    on_overlapping,
                    fields_breadcrumbs,
                )
            })?;
        }

        _ => {}
    }

    ControlFlow::Continue(())
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

impl<'tcx> LateLintPass<'tcx> for MutableTransmutes {
    fn check_expr(&mut self, cx: &LateContext<'_>, expr: &hir::Expr<'_>) {
        let Some((src, dst)) = get_transmute_from_to(cx, expr) else { return };

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
                    CollectFields::DoNotCheckFields
                }
                _ => CollectFields::CheckFields,
            },
            &mut FieldsBreadcrumbs::default(),
        );
        dst_refs.sort_unstable_by_key(|ref_| ref_.start_offset);
        check_overlapping(
            cx,
            &dst_refs,
            0,
            src,
            Size::ZERO,
            &mut |ty, idx, src_fields_breadcrumbs| {
                let dst_ref = &dst_refs[idx];
                match dst_ref.mutability {
                    // For mutable references in the target type, we need to see if they were converted from shared references.
                    Mutability::Mut => match ty.kind() {
                        ty::Ref(_, _, Mutability::Not) => {
                            let mut from = src.to_string();
                            src_fields_breadcrumbs.translate_for_diagnostics(cx, &mut from, src);
                            let mut to = dst.to_string();
                            dst_ref.fields_breadcrumbs.translate_for_diagnostics(cx, &mut to, dst);
                            cx.emit_spanned_lint(
                                MUTABLE_TRANSMUTES,
                                expr.span,
                                BuiltinMutablesTransmutes { from, to },
                            );
                            CheckOverlapping::ReportedError
                        }
                        _ => CheckOverlapping::CheckFields,
                    },
                    // For shared references, we need to see if they were transmuted from shared (not mutable!) references,
                    // and if there is an incorrectly transmuted `UnsafeCell` inside.
                    Mutability::Not => {
                        let &ty::Ref(_, src_ty, Mutability::Not) = ty.kind() else {
                            return CheckOverlapping::CheckFields;
                        };

                        let mut dst_unsafe_cells = Vec::new();
                        collect_fields(
                            cx,
                            dst_ref.ty,
                            Size::ZERO,
                            &mut |offset, ty, fields_breadcrumbs| match ty.kind() {
                                &ty::Adt(adt_def, _) if adt_def.is_unsafe_cell() => {
                                    let Ok(layout) = cx.layout_of(ty) else {
                                        return CollectFields::CheckFields;
                                    };
                                    if layout.is_zst() {
                                        return CollectFields::DoNotCheckFields;
                                    }

                                    let Ok(layout) = cx.layout_of(ty) else {
                                        return CollectFields::CheckFields;
                                    };
                                    dst_unsafe_cells.push(UnsafeCellField {
                                        start_offset: offset,
                                        end_offset: offset + layout.size,
                                        fields_breadcrumbs: fields_breadcrumbs.clone(),
                                    });
                                    CollectFields::DoNotCheckFields
                                }
                                _ => CollectFields::CheckFields,
                            },
                            &mut FieldsBreadcrumbs::default(),
                        );
                        dst_unsafe_cells
                            .sort_unstable_by_key(|unsafe_cell| unsafe_cell.start_offset);

                        let found_error = check_overlapping(
                            cx,
                            &dst_unsafe_cells,
                            0,
                            src_ty,
                            Size::ZERO,
                            &mut |ty, idx, src_pointee_fields_breadcrumbs| {
                                // First, check that there are actually some bytes there, not just padding or ZST.
                                match ty.kind() {
                                    // Transmuting `&UnsafeCell` to `&UnsafeCell` is fine, don't check inside.
                                    ty::Adt(adt_def, _) if adt_def.is_unsafe_cell() => {
                                        return CheckOverlapping::DoNotCheckFields;
                                    }

                                    ty::Bool
                                    | ty::Char
                                    | ty::Int(_)
                                    | ty::Uint(_)
                                    | ty::Float(_)
                                    | ty::RawPtr(_)
                                    | ty::Ref(_, _, _)
                                    | ty::FnPtr(_) => {}
                                    // Enums have their discriminant.
                                    ty::Adt(adt_def, _) if adt_def.is_enum() => {}
                                    _ => return CheckOverlapping::CheckFields,
                                }

                                let from = FieldsBreadcrumbs::unsafe_cell_breadcrumbs(
                                    cx,
                                    src_fields_breadcrumbs,
                                    src,
                                    src_pointee_fields_breadcrumbs,
                                    src_ty,
                                );
                                let to = FieldsBreadcrumbs::unsafe_cell_breadcrumbs(
                                    cx,
                                    &dst_ref.fields_breadcrumbs,
                                    dst,
                                    &dst_unsafe_cells[idx].fields_breadcrumbs,
                                    dst_ref.ty,
                                );

                                cx.emit_spanned_lint(
                                    UNSAFE_CELL_TRANSMUTES,
                                    expr.span,
                                    BuiltinUnsafeCellTransmutes { from, to },
                                );

                                CheckOverlapping::ReportedError
                            },
                            &mut FieldsBreadcrumbs::default(),
                        );
                        match found_error {
                            ControlFlow::Continue(()) => CheckOverlapping::CheckFields,
                            ControlFlow::Break(()) => CheckOverlapping::ReportedError,
                        }
                    }
                }
            },
            &mut FieldsBreadcrumbs::default(),
        );
    }
}

fn get_transmute_from_to<'tcx>(
    cx: &LateContext<'tcx>,
    expr: &hir::Expr<'_>,
) -> Option<(Ty<'tcx>, Ty<'tcx>)> {
    let def = if let hir::ExprKind::Path(ref qpath) = expr.kind {
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
    cx.tcx.is_intrinsic(def_id) && cx.tcx.item_name(def_id) == sym::transmute
}
