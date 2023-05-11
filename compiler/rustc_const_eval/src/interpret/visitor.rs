//! Visitor for a run-time value with a given layout: Traverse enums, structs and other compound
//! types until we arrive at the leaves, with custom handling for primitive types.

use rustc_middle::mir::interpret::InterpResult;
use rustc_middle::ty;
use rustc_middle::ty::layout::TyAndLayout;
use rustc_target::abi::{FieldsShape, VariantIdx, Variants};

use std::num::NonZeroUsize;

use super::{InterpCx, MPlaceTy, Machine, OpTy, PlaceTy};

/// A thing that we can project into, and that has a layout.
/// This wouldn't have to depend on `Machine` but with the current type inference,
/// that's just more convenient to work with (avoids repeating all the `Machine` bounds).
pub trait Value<'mir, 'tcx, M: Machine<'mir, 'tcx>>: Sized {
    /// Gets this value's layout.
    fn layout(&self) -> TyAndLayout<'tcx>;

    /// Makes this into an `OpTy`, in a cheap way that is good for reading.
    fn to_op_for_read(
        &self,
        ecx: &InterpCx<'mir, 'tcx, M>,
    ) -> InterpResult<'tcx, OpTy<'tcx, M::Provenance>>;

    /// Makes this into an `OpTy`, in a potentially more expensive way that is good for projections.
    fn to_op_for_proj(
        &self,
        ecx: &InterpCx<'mir, 'tcx, M>,
    ) -> InterpResult<'tcx, OpTy<'tcx, M::Provenance>> {
        self.to_op_for_read(ecx)
    }

    /// Creates this from an `OpTy`.
    ///
    /// If `to_op_for_proj` only ever produces `Indirect` operands, then this one is definitely `Indirect`.
    fn from_op(op: &OpTy<'tcx, M::Provenance>) -> Self;

    /// Projects to the given enum variant.
    fn project_downcast(
        &self,
        ecx: &InterpCx<'mir, 'tcx, M>,
        variant: VariantIdx,
    ) -> InterpResult<'tcx, Self>;

    /// Projects to the n-th field.
    fn project_field(
        &self,
        ecx: &InterpCx<'mir, 'tcx, M>,
        field: usize,
    ) -> InterpResult<'tcx, Self>;
}

/// A thing that we can project into given *mutable* access to `ecx`, and that has a layout.
/// This wouldn't have to depend on `Machine` but with the current type inference,
/// that's just more convenient to work with (avoids repeating all the `Machine` bounds).
pub trait ValueMut<'mir, 'tcx, M: Machine<'mir, 'tcx>>: Sized {
    /// Gets this value's layout.
    fn layout(&self) -> TyAndLayout<'tcx>;

    /// Makes this into an `OpTy`, in a cheap way that is good for reading.
    fn to_op_for_read(
        &self,
        ecx: &InterpCx<'mir, 'tcx, M>,
    ) -> InterpResult<'tcx, OpTy<'tcx, M::Provenance>>;

    /// Makes this into an `OpTy`, in a potentially more expensive way that is good for projections.
    fn to_op_for_proj(
        &self,
        ecx: &mut InterpCx<'mir, 'tcx, M>,
    ) -> InterpResult<'tcx, OpTy<'tcx, M::Provenance>>;

    /// Creates this from an `OpTy`.
    ///
    /// If `to_op_for_proj` only ever produces `Indirect` operands, then this one is definitely `Indirect`.
    fn from_op(op: &OpTy<'tcx, M::Provenance>) -> Self;

    /// Projects to the given enum variant.
    fn project_downcast(
        &self,
        ecx: &mut InterpCx<'mir, 'tcx, M>,
        variant: VariantIdx,
    ) -> InterpResult<'tcx, Self>;

    /// Projects to the n-th field.
    fn project_field(
        &self,
        ecx: &mut InterpCx<'mir, 'tcx, M>,
        field: usize,
    ) -> InterpResult<'tcx, Self>;
}

// We cannot have a general impl which shows that Value implies ValueMut. (When we do, it says we
// cannot `impl ValueMut for PlaceTy` because some downstream crate could `impl Value for PlaceTy`.)
// So we have some copy-paste here. (We could have a macro but since we only have 2 types with this
// double-impl, that would barely make the code shorter, if at all.)

impl<'mir, 'tcx: 'mir, M: Machine<'mir, 'tcx>> Value<'mir, 'tcx, M> for OpTy<'tcx, M::Provenance> {
    #[inline(always)]
    fn layout(&self) -> TyAndLayout<'tcx> {
        self.layout
    }

    #[inline(always)]
    fn to_op_for_read(
        &self,
        _ecx: &InterpCx<'mir, 'tcx, M>,
    ) -> InterpResult<'tcx, OpTy<'tcx, M::Provenance>> {
        Ok(self.clone())
    }

    #[inline(always)]
    fn from_op(op: &OpTy<'tcx, M::Provenance>) -> Self {
        op.clone()
    }

    #[inline(always)]
    fn project_downcast(
        &self,
        ecx: &InterpCx<'mir, 'tcx, M>,
        variant: VariantIdx,
    ) -> InterpResult<'tcx, Self> {
        ecx.operand_downcast(self, variant)
    }

    #[inline(always)]
    fn project_field(
        &self,
        ecx: &InterpCx<'mir, 'tcx, M>,
        field: usize,
    ) -> InterpResult<'tcx, Self> {
        ecx.operand_field(self, field)
    }
}

impl<'mir, 'tcx: 'mir, M: Machine<'mir, 'tcx>> ValueMut<'mir, 'tcx, M>
    for OpTy<'tcx, M::Provenance>
{
    #[inline(always)]
    fn layout(&self) -> TyAndLayout<'tcx> {
        self.layout
    }

    #[inline(always)]
    fn to_op_for_read(
        &self,
        _ecx: &InterpCx<'mir, 'tcx, M>,
    ) -> InterpResult<'tcx, OpTy<'tcx, M::Provenance>> {
        Ok(self.clone())
    }

    #[inline(always)]
    fn to_op_for_proj(
        &self,
        _ecx: &mut InterpCx<'mir, 'tcx, M>,
    ) -> InterpResult<'tcx, OpTy<'tcx, M::Provenance>> {
        Ok(self.clone())
    }

    #[inline(always)]
    fn from_op(op: &OpTy<'tcx, M::Provenance>) -> Self {
        op.clone()
    }

    #[inline(always)]
    fn project_downcast(
        &self,
        ecx: &mut InterpCx<'mir, 'tcx, M>,
        variant: VariantIdx,
    ) -> InterpResult<'tcx, Self> {
        ecx.operand_downcast(self, variant)
    }

    #[inline(always)]
    fn project_field(
        &self,
        ecx: &mut InterpCx<'mir, 'tcx, M>,
        field: usize,
    ) -> InterpResult<'tcx, Self> {
        ecx.operand_field(self, field)
    }
}

impl<'mir, 'tcx: 'mir, M: Machine<'mir, 'tcx>> Value<'mir, 'tcx, M>
    for MPlaceTy<'tcx, M::Provenance>
{
    #[inline(always)]
    fn layout(&self) -> TyAndLayout<'tcx> {
        self.layout
    }

    #[inline(always)]
    fn to_op_for_read(
        &self,
        _ecx: &InterpCx<'mir, 'tcx, M>,
    ) -> InterpResult<'tcx, OpTy<'tcx, M::Provenance>> {
        Ok(self.into())
    }

    #[inline(always)]
    fn from_op(op: &OpTy<'tcx, M::Provenance>) -> Self {
        // assert is justified because our `to_op_for_read` only ever produces `Indirect` operands.
        op.assert_mem_place()
    }

    #[inline(always)]
    fn project_downcast(
        &self,
        ecx: &InterpCx<'mir, 'tcx, M>,
        variant: VariantIdx,
    ) -> InterpResult<'tcx, Self> {
        ecx.mplace_downcast(self, variant)
    }

    #[inline(always)]
    fn project_field(
        &self,
        ecx: &InterpCx<'mir, 'tcx, M>,
        field: usize,
    ) -> InterpResult<'tcx, Self> {
        ecx.mplace_field(self, field)
    }
}

impl<'mir, 'tcx: 'mir, M: Machine<'mir, 'tcx>> ValueMut<'mir, 'tcx, M>
    for MPlaceTy<'tcx, M::Provenance>
{
    #[inline(always)]
    fn layout(&self) -> TyAndLayout<'tcx> {
        self.layout
    }

    #[inline(always)]
    fn to_op_for_read(
        &self,
        _ecx: &InterpCx<'mir, 'tcx, M>,
    ) -> InterpResult<'tcx, OpTy<'tcx, M::Provenance>> {
        Ok(self.into())
    }

    #[inline(always)]
    fn to_op_for_proj(
        &self,
        _ecx: &mut InterpCx<'mir, 'tcx, M>,
    ) -> InterpResult<'tcx, OpTy<'tcx, M::Provenance>> {
        Ok(self.into())
    }

    #[inline(always)]
    fn from_op(op: &OpTy<'tcx, M::Provenance>) -> Self {
        // assert is justified because our `to_op_for_proj` only ever produces `Indirect` operands.
        op.assert_mem_place()
    }

    #[inline(always)]
    fn project_downcast(
        &self,
        ecx: &mut InterpCx<'mir, 'tcx, M>,
        variant: VariantIdx,
    ) -> InterpResult<'tcx, Self> {
        ecx.mplace_downcast(self, variant)
    }

    #[inline(always)]
    fn project_field(
        &self,
        ecx: &mut InterpCx<'mir, 'tcx, M>,
        field: usize,
    ) -> InterpResult<'tcx, Self> {
        ecx.mplace_field(self, field)
    }
}

impl<'mir, 'tcx: 'mir, M: Machine<'mir, 'tcx>> ValueMut<'mir, 'tcx, M>
    for PlaceTy<'tcx, M::Provenance>
{
    #[inline(always)]
    fn layout(&self) -> TyAndLayout<'tcx> {
        self.layout
    }

    #[inline(always)]
    fn to_op_for_read(
        &self,
        ecx: &InterpCx<'mir, 'tcx, M>,
    ) -> InterpResult<'tcx, OpTy<'tcx, M::Provenance>> {
        // No need for `force_allocation` since we are just going to read from this.
        ecx.place_to_op(self)
    }

    #[inline(always)]
    fn to_op_for_proj(
        &self,
        ecx: &mut InterpCx<'mir, 'tcx, M>,
    ) -> InterpResult<'tcx, OpTy<'tcx, M::Provenance>> {
        // We `force_allocation` here so that `from_op` below can work.
        Ok(ecx.force_allocation(self)?.into())
    }

    #[inline(always)]
    fn from_op(op: &OpTy<'tcx, M::Provenance>) -> Self {
        // assert is justified because our `to_op` only ever produces `Indirect` operands.
        op.assert_mem_place().into()
    }

    #[inline(always)]
    fn project_downcast(
        &self,
        ecx: &mut InterpCx<'mir, 'tcx, M>,
        variant: VariantIdx,
    ) -> InterpResult<'tcx, Self> {
        ecx.place_downcast(self, variant)
    }

    #[inline(always)]
    fn project_field(
        &self,
        ecx: &mut InterpCx<'mir, 'tcx, M>,
        field: usize,
    ) -> InterpResult<'tcx, Self> {
        ecx.place_field(self, field)
    }
}

macro_rules! make_value_visitor {
    ($visitor_trait:ident, $value_trait:ident, $($mutability:ident)?) => {
        /// How to traverse a value and what to do when we are at the leaves.
        pub trait $visitor_trait<'mir, 'tcx: 'mir, M: Machine<'mir, 'tcx>>: Sized {
            type V: $value_trait<'mir, 'tcx, M>;

            /// The visitor must have an `InterpCx` in it.
            fn ecx(&$($mutability)? self)
                -> &$($mutability)? InterpCx<'mir, 'tcx, M>;

            /// `read_discriminant` can be hooked for better error messages.
            #[inline(always)]
            fn read_discriminant(
                &mut self,
                op: &OpTy<'tcx, M::Provenance>,
            ) -> InterpResult<'tcx, VariantIdx> {
                Ok(self.ecx().read_discriminant(op)?.1)
            }

            // Recursive actions, ready to be overloaded.
            /// Visits the given value, dispatching as appropriate to more specialized visitors.
            #[inline(always)]
            fn visit_value(&mut self, v: &Self::V) -> InterpResult<'tcx>
            {
                self.walk_value(v)
            }
            /// Visits the given value as a union. No automatic recursion can happen here.
            #[inline(always)]
            fn visit_union(&mut self, _v: &Self::V, _fields: NonZeroUsize) -> InterpResult<'tcx>
            {
                Ok(())
            }
            /// Visits the given value as the pointer of a `Box`. There is nothing to recurse into.
            /// The type of `v` will be a raw pointer, but this is a field of `Box<T>` and the
            /// pointee type is the actual `T`.
            #[inline(always)]
            fn visit_box(&mut self, _v: &Self::V) -> InterpResult<'tcx>
            {
                Ok(())
            }
            /// Visits this value as an aggregate, you are getting an iterator yielding
            /// all the fields (still in an `InterpResult`, you have to do error handling yourself).
            /// Recurses into the fields.
            #[inline(always)]
            fn visit_aggregate(
                &mut self,
                v: &Self::V,
                fields: impl Iterator<Item=InterpResult<'tcx, Self::V>>,
            ) -> InterpResult<'tcx> {
                self.walk_aggregate(v, fields)
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

            // Default recursors. Not meant to be overloaded.
            fn walk_aggregate(
                &mut self,
                v: &Self::V,
                fields: impl Iterator<Item=InterpResult<'tcx, Self::V>>,
            ) -> InterpResult<'tcx> {
                // Now iterate over it.
                for (idx, field_val) in fields.enumerate() {
                    self.visit_field(v, idx, &field_val?)?;
                }
                Ok(())
            }
            fn walk_value(&mut self, v: &Self::V) -> InterpResult<'tcx>
            {
                let ty = v.layout().ty;
                trace!("walk_value: type: {ty}");

                // Special treatment for special types, where the (static) layout is not sufficient.
                match *ty.kind() {
                    // If it is a trait object, switch to the real type that was used to create it.
                    ty::Dynamic(_, _, ty::Dyn) => {
                        // Dyn types. This is unsized, and the actual dynamic type of the data is given by the
                        // vtable stored in the place metadata.
                        // unsized values are never immediate, so we can assert_mem_place
                        let op = v.to_op_for_read(self.ecx())?;
                        let dest = op.assert_mem_place();
                        let inner_mplace = self.ecx().unpack_dyn_trait(&dest)?.0;
                        trace!("walk_value: dyn object layout: {:#?}", inner_mplace.layout);
                        // recurse with the inner type
                        return self.visit_field(&v, 0, &$value_trait::from_op(&inner_mplace.into()));
                    },
                    ty::Dynamic(_, _, ty::DynStar) => {
                        // DynStar types. Very different from a dyn type (but strangely part of the
                        // same variant in `TyKind`): These are pairs where the 2nd component is the
                        // vtable, and the first component is the data (which must be ptr-sized).
                        let op = v.to_op_for_proj(self.ecx())?;
                        let data = self.ecx().unpack_dyn_star(&op)?.0;
                        return self.visit_field(&v, 0, &$value_trait::from_op(&data));
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
                        // When we hit a `Box`, we do not do the usual `visit_aggregate`; instead,
                        // we (a) call `visit_box` on the pointer value, and (b) recurse on the
                        // allocator field. We also assert tons of things to ensure we do not miss
                        // any other fields.

                        // `Box` has two fields: the pointer we care about, and the allocator.
                        assert_eq!(v.layout().fields.count(), 2, "`Box` must have exactly 2 fields");
                        let (unique_ptr, alloc) =
                            (v.project_field(self.ecx(), 0)?, v.project_field(self.ecx(), 1)?);
                        // Unfortunately there is some type junk in the way here: `unique_ptr` is a `Unique`...
                        // (which means another 2 fields, the second of which is a `PhantomData`)
                        assert_eq!(unique_ptr.layout().fields.count(), 2);
                        let (nonnull_ptr, phantom) = (
                            unique_ptr.project_field(self.ecx(), 0)?,
                            unique_ptr.project_field(self.ecx(), 1)?,
                        );
                        assert!(
                            phantom.layout().ty.ty_adt_def().is_some_and(|adt| adt.is_phantom_data()),
                            "2nd field of `Unique` should be PhantomData but is {:?}",
                            phantom.layout().ty,
                        );
                        // ... that contains a `NonNull`... (gladly, only a single field here)
                        assert_eq!(nonnull_ptr.layout().fields.count(), 1);
                        let raw_ptr = nonnull_ptr.project_field(self.ecx(), 0)?; // the actual raw ptr
                        // ... whose only field finally is a raw ptr we can dereference.
                        self.visit_box(&raw_ptr)?;

                        // The second `Box` field is the allocator, which we recursively check for validity
                        // like in regular structs.
                        self.visit_field(v, 1, &alloc)?;

                        // We visited all parts of this one.
                        return Ok(());
                    }
                    _ => {},
                };

                // Visit the fields of this value.
                match &v.layout().fields {
                    FieldsShape::Primitive => {}
                    &FieldsShape::Union(fields) => {
                        self.visit_union(v, fields)?;
                    }
                    FieldsShape::Arbitrary { offsets, .. } => {
                        // FIXME: We collect in a vec because otherwise there are lifetime
                        // errors: Projecting to a field needs access to `ecx`.
                        let fields: Vec<InterpResult<'tcx, Self::V>> =
                            (0..offsets.len()).map(|i| {
                                v.project_field(self.ecx(), i)
                            })
                            .collect();
                        self.visit_aggregate(v, fields.into_iter())?;
                    }
                    FieldsShape::Array { .. } => {
                        // Let's get an mplace (or immediate) first.
                        // This might `force_allocate` if `v` is a `PlaceTy`, but `place_index` does that anyway.
                        let op = v.to_op_for_proj(self.ecx())?;
                        // Now we can go over all the fields.
                        // This uses the *run-time length*, i.e., if we are a slice,
                        // the dynamic info from the metadata is used.
                        let iter = self.ecx().operand_array_fields(&op)?
                            .map(|f| f.and_then(|f| {
                                Ok($value_trait::from_op(&f))
                            }));
                        self.visit_aggregate(v, iter)?;
                    }
                }

                match v.layout().variants {
                    // If this is a multi-variant layout, find the right variant and proceed
                    // with *its* fields.
                    Variants::Multiple { .. } => {
                        let op = v.to_op_for_read(self.ecx())?;
                        let idx = self.read_discriminant(&op)?;
                        let inner = v.project_downcast(self.ecx(), idx)?;
                        trace!("walk_value: variant layout: {:#?}", inner.layout());
                        // recurse with the inner type
                        self.visit_variant(v, idx, &inner)
                    }
                    // For single-variant layouts, we already did anything there is to do.
                    Variants::Single { .. } => Ok(())
                }
            }
        }
    }
}

make_value_visitor!(ValueVisitor, Value,);
make_value_visitor!(MutValueVisitor, ValueMut, mut);
