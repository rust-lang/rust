//! Visitor for a run-time value with a given layout: Traverse enums, structs and other compound
//! types until we arrive at the leaves, with custom handling for primitive types.

use std::num::NonZero;

use rustc_abi::{FieldIdx, FieldsShape, VariantIdx, Variants};
use rustc_index::IndexVec;
use rustc_middle::mir::interpret::InterpResult;
use rustc_middle::ty::{self, Ty};
use tracing::trace;

use super::{InterpCx, MPlaceTy, Machine, Projectable, interp_ok, throw_inval};

/// How to traverse a value and what to do when we are at the leaves.
pub trait ValueVisitor<'tcx, M: Machine<'tcx>>: Sized {
    type V: Projectable<'tcx, M::Provenance> + From<MPlaceTy<'tcx, M::Provenance>>;

    /// The visitor must have an `InterpCx` in it.
    fn ecx(&self) -> &InterpCx<'tcx, M>;

    /// `read_discriminant` can be hooked for better error messages.
    #[inline(always)]
    fn read_discriminant(&mut self, v: &Self::V) -> InterpResult<'tcx, VariantIdx> {
        self.ecx().read_discriminant(&v.to_op(self.ecx())?)
    }

    /// This function provides the chance to reorder the order in which fields are visited for
    /// `FieldsShape::Aggregate`.
    ///
    /// The default means we iterate in source declaration order; alternatively this can do some
    /// work with `memory_index` to iterate in memory order.
    #[inline(always)]
    fn aggregate_field_iter(
        memory_index: &IndexVec<FieldIdx, u32>,
    ) -> impl Iterator<Item = FieldIdx> + 'static {
        memory_index.indices()
    }

    // Recursive actions, ready to be overloaded.
    /// Visits the given value, dispatching as appropriate to more specialized visitors.
    #[inline(always)]
    fn visit_value(&mut self, v: &Self::V) -> InterpResult<'tcx> {
        self.walk_value(v)
    }
    /// Visits the given value as a union. No automatic recursion can happen here.
    #[inline(always)]
    fn visit_union(&mut self, _v: &Self::V, _fields: NonZero<usize>) -> InterpResult<'tcx> {
        interp_ok(())
    }
    /// Visits the given value as the pointer of a `Box`. There is nothing to recurse into.
    /// The type of `v` will be a raw pointer to `T`, but this is a field of `Box<T>` and the
    /// pointee type is the actual `T`. `box_ty` provides the full type of the `Box` itself.
    #[inline(always)]
    fn visit_box(&mut self, _box_ty: Ty<'tcx>, _v: &Self::V) -> InterpResult<'tcx> {
        interp_ok(())
    }

    /// Called each time we recurse down to a field of a "product-like" aggregate
    /// (structs, tuples, arrays and the like, but not enums), passing in old (outer)
    /// and new (inner) value.
    /// This gives the visitor the chance to track the stack of nested fields that
    /// we are descending through.
    #[inline(always)]
    fn visit_field(
        &mut self,
        _old_val: &Self::V,
        _field: usize,
        new_val: &Self::V,
    ) -> InterpResult<'tcx> {
        self.visit_value(new_val)
    }
    /// Called when recursing into an enum variant.
    /// This gives the visitor the chance to track the stack of nested fields that
    /// we are descending through.
    #[inline(always)]
    fn visit_variant(
        &mut self,
        _old_val: &Self::V,
        _variant: VariantIdx,
        new_val: &Self::V,
    ) -> InterpResult<'tcx> {
        self.visit_value(new_val)
    }

    /// Traversal logic; should not be overloaded.
    fn walk_value(&mut self, v: &Self::V) -> InterpResult<'tcx> {
        let ty = v.layout().ty;
        trace!("walk_value: type: {ty}");

        // Special treatment for special types, where the (static) layout is not sufficient.
        match *ty.kind() {
            // If it is a trait object, switch to the real type that was used to create it.
            ty::Dynamic(data, _, ty::Dyn) => {
                // Dyn types. This is unsized, and the actual dynamic type of the data is given by the
                // vtable stored in the place metadata.
                // unsized values are never immediate, so we can assert_mem_place
                let op = v.to_op(self.ecx())?;
                let dest = op.assert_mem_place();
                let inner_mplace = self.ecx().unpack_dyn_trait(&dest, data)?;
                trace!("walk_value: dyn object layout: {:#?}", inner_mplace.layout);
                // recurse with the inner type
                return self.visit_field(v, 0, &inner_mplace.into());
            }
            // Slices do not need special handling here: they have `Array` field
            // placement with length 0, so we enter the `Array` case below which
            // indirectly uses the metadata to determine the actual length.

            // However, `Box`... let's talk about `Box`.
            ty::Adt(def, ..) if def.is_box() => {
                // `Box` is a hybrid primitive-library-defined type that one the one hand is
                // a dereferenceable pointer, on the other hand has *basically arbitrary
                // user-defined layout* since the user controls the 'allocator' field. So it
                // cannot be treated like a normal pointer, since it does not fit into an
                // `Immediate`. Yeah, it is quite terrible. But many visitors want to do
                // something with "all boxed pointers", so we handle this mess for them.
                //
                // When we hit a `Box`, we do not do the usual field recursion; instead,
                // we (a) call `visit_box` on the pointer value, and (b) recurse on the
                // allocator field. We also assert tons of things to ensure we do not miss
                // any other fields.

                // `Box` has two fields: the pointer we care about, and the allocator.
                assert_eq!(v.layout().fields.count(), 2, "`Box` must have exactly 2 fields");
                let (unique_ptr, alloc) = (
                    self.ecx().project_field(v, FieldIdx::ZERO)?,
                    self.ecx().project_field(v, FieldIdx::ONE)?,
                );
                // Unfortunately there is some type junk in the way here: `unique_ptr` is a `Unique`...
                // (which means another 2 fields, the second of which is a `PhantomData`)
                assert_eq!(unique_ptr.layout().fields.count(), 2);
                let (nonnull_ptr, phantom) = (
                    self.ecx().project_field(&unique_ptr, FieldIdx::ZERO)?,
                    self.ecx().project_field(&unique_ptr, FieldIdx::ONE)?,
                );
                assert!(
                    phantom.layout().ty.ty_adt_def().is_some_and(|adt| adt.is_phantom_data()),
                    "2nd field of `Unique` should be PhantomData but is {:?}",
                    phantom.layout().ty,
                );
                // ... that contains a `NonNull`... (gladly, only a single field here)
                assert_eq!(nonnull_ptr.layout().fields.count(), 1);
                let raw_ptr = self.ecx().project_field(&nonnull_ptr, FieldIdx::ZERO)?; // the actual raw ptr
                // ... whose only field finally is a raw ptr we can dereference.
                self.visit_box(ty, &raw_ptr)?;

                // The second `Box` field is the allocator, which we recursively check for validity
                // like in regular structs.
                self.visit_field(v, 1, &alloc)?;

                // We visited all parts of this one.
                return interp_ok(());
            }

            // Non-normalized types should never show up here.
            ty::Param(..)
            | ty::Alias(..)
            | ty::Bound(..)
            | ty::Placeholder(..)
            | ty::Infer(..)
            | ty::Error(..) => throw_inval!(TooGeneric),

            // The rest is handled below.
            _ => {}
        };

        // Visit the fields of this value.
        match &v.layout().fields {
            FieldsShape::Primitive => {}
            &FieldsShape::Union(fields) => {
                self.visit_union(v, fields)?;
            }
            FieldsShape::Arbitrary { memory_index, .. } => {
                for idx in Self::aggregate_field_iter(memory_index) {
                    let field = self.ecx().project_field(v, idx)?;
                    self.visit_field(v, idx.as_usize(), &field)?;
                }
            }
            FieldsShape::Array { .. } => {
                let mut iter = self.ecx().project_array_fields(v)?;
                while let Some((idx, field)) = iter.next(self.ecx())? {
                    self.visit_field(v, idx.try_into().unwrap(), &field)?;
                }
            }
        }

        match v.layout().variants {
            // If this is a multi-variant layout, find the right variant and proceed
            // with *its* fields.
            Variants::Multiple { .. } => {
                let idx = self.read_discriminant(v)?;
                // There are 3 cases where downcasts can turn a Scalar/ScalarPair into a different ABI which
                // could be a problem for `ImmTy` (see layout_sanity_check):
                // - variant.size == Size::ZERO: works fine because `ImmTy::offset` has a special case for
                //   zero-sized layouts.
                // - variant.fields.count() == 0: works fine because `ImmTy::offset` has a special case for
                //   zero-field aggregates.
                // - variant.abi.is_uninhabited(): triggers UB in `read_discriminant` so we never get here.
                let inner = self.ecx().project_downcast(v, idx)?;
                trace!("walk_value: variant layout: {:#?}", inner.layout());
                // recurse with the inner type
                self.visit_variant(v, idx, &inner)?;
            }
            // For single-variant layouts, we already did everything there is to do.
            Variants::Single { .. } | Variants::Empty => {}
        }

        interp_ok(())
    }
}
