//! Experimental support for emitting retags as function calls in generated code.
//!
//! We attempt to retag every argument and return value of a function, and every rvalue
//! of an assignment. The first step to retagging is to generate a [`RetagPlan`], which
//! describes which pointers within the place or operand can be retagged.

#![allow(unused)]
use rustc_abi::{BackendRepr, FieldIdx, FieldsShape, VariantIdx, Variants};
use rustc_ast::Mutability;
use rustc_data_structures::fx::FxIndexMap;
use rustc_middle::mir::{Rvalue, WithRetag};
use rustc_middle::ty;
use rustc_middle::ty::layout::TyAndLayout;

use crate::RetagInfo;
use crate::mir::FunctionCx;
use crate::mir::operand::OperandRef;
use crate::mir::place::PlaceRef;
use crate::traits::BuilderMethods;

pub(crate) fn rvalue_needs_retag(rvalue: &Rvalue<'_>) -> bool {
    // `Ref` has its own internal retagging
    !matches!(rvalue, Rvalue::Ref(..)) && !matches!(rvalue, Rvalue::Use(.., WithRetag::No))
}

/// A description of the pointers within a type that need to be retagged.
#[derive(Debug)]
enum RetagPlan<V> {
    /// Indicates that a pointer should be retagged.
    EmitRetag(RetagInfo<V>),
    /// Indicates that one or more fields or variants of this type
    /// contain pointers that need to be retagged.
    Recurse {
        field_plans: FxIndexMap<FieldIdx, RetagPlan<V>>,
        variant_plans: FxIndexMap<VariantIdx, RetagPlan<V>>,
    },
}

impl<V> RetagPlan<V> {
    /// A helper function to move a [`RetagPlan`] into a particular field.
    fn for_field(self, ix: FieldIdx) -> Self {
        let mut field_plans = FxIndexMap::default();
        field_plans.insert(ix, self);
        RetagPlan::Recurse { field_plans, variant_plans: FxIndexMap::default() }
    }
}

impl<'a, 'tcx, V> RetagPlan<V> {
    /// Attempts to create a [`RetagPlan`] for a place or operand with the given layout.
    fn build<Bx: BuilderMethods<'a, 'tcx>>(
        bx: &mut Bx,
        layout: TyAndLayout<'tcx>,
        is_fn_entry: bool,
    ) -> Option<RetagPlan<Bx::Value>> {
        // If the value being retagged is smaller than a pointer, then it can't contain any
        // pointers we need to retag, so we can stop recursion early. This optimization is
        // crucial for ZSTs, because they can contain way more fields than we can ever visit.
        if layout.is_sized() && layout.size < bx.tcx().data_layout.pointer_size() {
            return None;
        }
        // SIMD vectors may only contain raw pointers, integers, and floating point values,
        // which do not need to be retagged.
        if matches!(layout.backend_repr, BackendRepr::SimdVector { .. }) {
            return None;
        }

        // Check the type of this value to see what to do with it (retag, or recurse).
        match layout.ty.kind() {
            &ty::Ref(_, pointee, mt) => {
                let pointee_layout = bx.layout_of(pointee);
                Self::emit_retag(bx, pointee_layout, Some(mt), is_fn_entry)
            }
            &ty::RawPtr(_, _) => None,
            // `Box` needs special handling, since the innermost pointer is what gets retagged, but
            //  the outermost `Box` is what determines the permission that gets created.
            ty::Adt(adt, _) if adt.is_box() => Self::visit_box(bx, layout, is_fn_entry),
            // Skip traversing for everything inside of `MaybeDangling`
            ty::Adt(adt, _) if adt.is_maybe_dangling() => None,
            _ => Self::walk_value(bx, layout, is_fn_entry),
        }
    }

    /// Recurses through the fields and variants of a value in memory order to create a [`RetagPlan`].
    fn walk_value<Bx: BuilderMethods<'a, 'tcx>>(
        bx: &mut Bx,
        layout: TyAndLayout<'tcx>,
        is_fn_entry: bool,
    ) -> Option<RetagPlan<Bx::Value>> {
        let mut field_plans = FxIndexMap::default();
        let mut variant_plans = FxIndexMap::default();

        match &layout.fields {
            FieldsShape::Union(_) | FieldsShape::Primitive => {}
            _ => {
                for ix in layout.fields.index_by_increasing_offset() {
                    let field_layout = layout.field(bx, ix);
                    if let Some(plan) = Self::build(bx, field_layout, is_fn_entry) {
                        field_plans.insert(FieldIdx::from_usize(ix), plan);
                    }
                }
            }
        }

        match &layout.variants {
            Variants::Single { .. } | Variants::Empty => {}
            Variants::Multiple { variants, .. } => {
                for ix in variants.indices() {
                    let variant_layout = layout.for_variant(bx, ix);
                    if let Some(plan) = Self::build(bx, variant_layout, is_fn_entry) {
                        variant_plans.insert(ix, plan);
                    }
                }
            }
        }

        (!field_plans.is_empty() || !variant_plans.is_empty())
            .then(|| RetagPlan::Recurse { field_plans, variant_plans })
    }

    /// Emits a retag for a `Box`.
    fn visit_box<Bx: BuilderMethods<'a, 'tcx>>(
        bx: &mut Bx,
        layout: TyAndLayout<'tcx>,
        is_fn_entry: bool,
    ) -> Option<RetagPlan<Bx::Value>> {
        assert!(layout.ty.is_box());
        assert_eq!(layout.fields.count(), 2, "`Box` must have exactly 2 fields");
        let mut field_plans = FxIndexMap::default();

        // Only retag the inner pointer of a `Box` if it came from the global allocator.
        if layout.ty.is_box_global(bx.tcx()) {
            let boxed_ty = layout.ty.expect_boxed_ty();
            let boxed_layout = bx.layout_of(boxed_ty);
            if let Some(mut plan) = Self::emit_retag(bx, boxed_layout, None, is_fn_entry) {
                // `Unique<T>`
                let unique = layout.field(bx, 0);
                assert_eq!(unique.fields.count(), 2);
                plan = plan.for_field(FieldIdx::ZERO);

                // `NonNull<T>`
                let nonnull = unique.field(bx, 0);
                assert_eq!(nonnull.fields.count(), 1);
                plan = plan.for_field(FieldIdx::ZERO);

                // `*mut T is !null`
                let pattern = nonnull.field(bx, 0);
                let ty::Pat(base, _) = pattern.ty.kind() else {
                    unreachable!("`NonNull` should contain a pattern type")
                };
                assert_eq!(base.builtin_deref(true), Some(boxed_ty));

                field_plans.insert(FieldIdx::ZERO, plan);
            }
        }

        // We always try to retag the second field (the allocator)
        let field_layout = layout.field(bx, 1);
        if let Some(plan) = Self::build(bx, field_layout, is_fn_entry) {
            field_plans.insert(FieldIdx::ONE, plan);
        }

        (!field_plans.is_empty())
            .then(|| RetagPlan::Recurse { field_plans, variant_plans: FxIndexMap::default() })
    }

    /// Determines if a pointer needs to be retagged, when it points to
    /// a type with the given layout. Returns `None` for mutable pointers
    /// to types that are entirely covered by `UnsafePinned`, for which retags
    /// are a no-op.
    fn emit_retag<Bx: BuilderMethods<'a, 'tcx>>(
        _bx: &mut Bx,
        _pointee_layout: TyAndLayout<'tcx>,
        _ptr_kind: Option<Mutability>,
        _is_fn_entry: bool,
    ) -> Option<RetagPlan<Bx::Value>> {
        None
    }
}

impl<'a, 'tcx, Bx: BuilderMethods<'a, 'tcx>> FunctionCx<'a, 'tcx, Bx> {
    /// Retags the pointers within an [`OperandRef`].
    pub(crate) fn codegen_retag_operand(
        &mut self,
        _bx: &mut Bx,
        operand: OperandRef<'tcx, Bx::Value>,
        _is_fn_entry: bool,
    ) -> OperandRef<'tcx, Bx::Value> {
        operand
    }

    /// Retags the pointers within a [`PlaceRef`].
    pub(crate) fn codegen_retag_place(
        &mut self,
        _bx: &mut Bx,
        _place_ref: PlaceRef<'tcx, Bx::Value>,
        _is_fn_entry: bool,
    ) {
    }
}
