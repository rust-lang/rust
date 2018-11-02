//! Visitor for a run-time value with a given layout: Traverse enums, structs and other compound
//! types until we arrive at the leaves, with custom handling for primitive types.

use rustc::ty::layout::{self, TyLayout};
use rustc::ty;
use rustc::mir::interpret::{
    EvalResult,
};

use super::{
    Machine, EvalContext, MPlaceTy, PlaceTy, OpTy,
};

// A thing that we can project into, and that has a layout.
// This wouldn't have to depend on `Machine` but with the current type inference,
// that's just more convenient to work with (avoids repeading all the `Machine` bounds).
pub trait Value<'a, 'mir, 'tcx, M: Machine<'a, 'mir, 'tcx>>: Copy
{
    // Get this value's layout.
    fn layout(&self) -> TyLayout<'tcx>;

    // Make this a `MPlaceTy`, or panic if that's not possible.
    fn to_mem_place(
        self,
        ectx: &EvalContext<'a, 'mir, 'tcx, M>,
    ) -> EvalResult<'tcx, MPlaceTy<'tcx, M::PointerTag>>;

    // Create this from an `MPlaceTy`.
    fn from_mem_place(MPlaceTy<'tcx, M::PointerTag>) -> Self;

    // Read the current enum discriminant, and downcast to that.  Also return the
    // variant index.
    fn project_downcast(
        self,
        ectx: &EvalContext<'a, 'mir, 'tcx, M>
    ) -> EvalResult<'tcx, (Self, usize)>;

    // Project to the n-th field.
    fn project_field(
        self,
        ectx: &mut EvalContext<'a, 'mir, 'tcx, M>,
        field: u64,
    ) -> EvalResult<'tcx, Self>;
}

// Operands and places are both values
impl<'a, 'mir, 'tcx, M: Machine<'a, 'mir, 'tcx>> Value<'a, 'mir, 'tcx, M>
    for OpTy<'tcx, M::PointerTag>
{
    #[inline(always)]
    fn layout(&self) -> TyLayout<'tcx> {
        self.layout
    }

    #[inline(always)]
    fn to_mem_place(
        self,
        _ectx: &EvalContext<'a, 'mir, 'tcx, M>,
    ) -> EvalResult<'tcx, MPlaceTy<'tcx, M::PointerTag>> {
        Ok(self.to_mem_place())
    }

    #[inline(always)]
    fn from_mem_place(mplace: MPlaceTy<'tcx, M::PointerTag>) -> Self {
        mplace.into()
    }

    #[inline(always)]
    fn project_downcast(
        self,
        ectx: &EvalContext<'a, 'mir, 'tcx, M>
    ) -> EvalResult<'tcx, (Self, usize)> {
        let idx = ectx.read_discriminant(self)?.1;
        Ok((ectx.operand_downcast(self, idx)?, idx))
    }

    #[inline(always)]
    fn project_field(
        self,
        ectx: &mut EvalContext<'a, 'mir, 'tcx, M>,
        field: u64,
    ) -> EvalResult<'tcx, Self> {
        ectx.operand_field(self, field)
    }
}
impl<'a, 'mir, 'tcx, M: Machine<'a, 'mir, 'tcx>> Value<'a, 'mir, 'tcx, M>
    for MPlaceTy<'tcx, M::PointerTag>
{
    #[inline(always)]
    fn layout(&self) -> TyLayout<'tcx> {
        self.layout
    }

    #[inline(always)]
    fn to_mem_place(
        self,
        _ectx: &EvalContext<'a, 'mir, 'tcx, M>,
    ) -> EvalResult<'tcx, MPlaceTy<'tcx, M::PointerTag>> {
        Ok(self)
    }

    #[inline(always)]
    fn from_mem_place(mplace: MPlaceTy<'tcx, M::PointerTag>) -> Self {
        mplace
    }

    #[inline(always)]
    fn project_downcast(
        self,
        ectx: &EvalContext<'a, 'mir, 'tcx, M>
    ) -> EvalResult<'tcx, (Self, usize)> {
        let idx = ectx.read_discriminant(self.into())?.1;
        Ok((ectx.mplace_downcast(self, idx)?, idx))
    }

    #[inline(always)]
    fn project_field(
        self,
        ectx: &mut EvalContext<'a, 'mir, 'tcx, M>,
        field: u64,
    ) -> EvalResult<'tcx, Self> {
        ectx.mplace_field(self, field)
    }
}
impl<'a, 'mir, 'tcx, M: Machine<'a, 'mir, 'tcx>> Value<'a, 'mir, 'tcx, M>
    for PlaceTy<'tcx, M::PointerTag>
{
    #[inline(always)]
    fn layout(&self) -> TyLayout<'tcx> {
        self.layout
    }

    #[inline(always)]
    fn to_mem_place(
        self,
        ectx: &EvalContext<'a, 'mir, 'tcx, M>,
    ) -> EvalResult<'tcx, MPlaceTy<'tcx, M::PointerTag>> {
        // If this refers to a local, assert that it already has an allocation.
        Ok(ectx.place_to_op(self)?.to_mem_place())
    }

    #[inline(always)]
    fn from_mem_place(mplace: MPlaceTy<'tcx, M::PointerTag>) -> Self {
        mplace.into()
    }

    #[inline(always)]
    fn project_downcast(
        self,
        ectx: &EvalContext<'a, 'mir, 'tcx, M>
    ) -> EvalResult<'tcx, (Self, usize)> {
        let idx = ectx.read_discriminant(ectx.place_to_op(self)?)?.1;
        Ok((ectx.place_downcast(self, idx)?, idx))
    }

    #[inline(always)]
    fn project_field(
        self,
        ectx: &mut EvalContext<'a, 'mir, 'tcx, M>,
        field: u64,
    ) -> EvalResult<'tcx, Self> {
        ectx.place_field(self, field)
    }
}

// How to traverse a value and what to do when we are at the leaves.
pub trait ValueVisitor<'a, 'mir, 'tcx: 'mir+'a, M: Machine<'a, 'mir, 'tcx>>: Sized {
    type V: Value<'a, 'mir, 'tcx, M>;

    // The visitor must have an `EvalContext` in it.
    fn ecx(&mut self) -> &mut EvalContext<'a, 'mir, 'tcx, M>;

    // Recursie actions, ready to be overloaded.
    /// Visit the given value, dispatching as appropriate to more speicalized visitors.
    #[inline(always)]
    fn visit_value(&mut self, v: Self::V) -> EvalResult<'tcx>
    {
        self.walk_value(v)
    }
    /// Visit the given value as a union.
    #[inline(always)]
    fn visit_union(&mut self, _v: Self::V) -> EvalResult<'tcx>
    {
        Ok(())
    }
    /// Visit the given value as an array.
    #[inline(always)]
    fn visit_array(&mut self, v: Self::V) -> EvalResult<'tcx>
    {
        self.walk_array(v)
    }
    /// Called each time we recurse down to a field, passing in old and new value.
    /// This gives the visitor the chance to track the stack of nested fields that
    /// we are descending through.
    #[inline(always)]
    fn visit_field(
        &mut self,
        _old_val: Self::V,
        _field: usize,
        new_val: Self::V,
    ) -> EvalResult<'tcx> {
        self.visit_value(new_val)
    }

    // Actions on the leaves, ready to be overloaded.
    /// Called whenever we reach a value with uninhabited layout.
    /// Recursing to fields will continue after this!
    #[inline(always)]
    fn visit_uninhabited(&mut self, _v: Self::V) -> EvalResult<'tcx>
    { Ok(()) }
    /// Called whenever we reach a value with scalar layout.
    /// Recursing to fields will continue after this!
    #[inline(always)]
    fn visit_scalar(&mut self, _v: Self::V, _layout: &layout::Scalar) -> EvalResult<'tcx>
    { Ok(()) }
    /// Called whenever we reach a value of primitive type.  There can be no recursion
    /// below such a value.
    #[inline(always)]
    fn visit_primitive(&mut self, _v: Self::V) -> EvalResult<'tcx>
    { Ok(()) }

    // Default recursors. Not meant to be overloaded.
    fn walk_array(&mut self, v: Self::V) -> EvalResult<'tcx>
    {
        // Let's get an mplace first.
        let mplace = if v.layout().is_zst() {
            // it's a ZST, the memory content cannot matter
            MPlaceTy::dangling(v.layout(), self.ecx())
        } else {
            // non-ZST array/slice/str cannot be immediate
            v.to_mem_place(self.ecx())?
        };
        // Now iterate over it.
        for (i, field) in self.ecx().mplace_array_fields(mplace)?.enumerate() {
            self.visit_field(v, i, Value::from_mem_place(field?))?;
        }
        Ok(())
    }
    fn walk_value(&mut self, v: Self::V) -> EvalResult<'tcx>
    {
        // If this is a multi-variant layout, we have find the right one and proceed with that.
        // (No benefit from making this recursion, but it is equivalent to that.)
        match v.layout().variants {
            layout::Variants::NicheFilling { .. } |
            layout::Variants::Tagged { .. } => {
                let (inner, idx) = v.project_downcast(self.ecx())?;
                trace!("variant layout: {:#?}", inner.layout());
                // recurse with the inner type
                return self.visit_field(v, idx, inner);
            }
            layout::Variants::Single { .. } => {}
        }

        // Even for single variants, we might be able to get a more refined type:
        // If it is a trait object, switch to the actual type that was used to create it.
        match v.layout().ty.sty {
            ty::Dynamic(..) => {
                // immediate trait objects are not a thing
                let dest = v.to_mem_place(self.ecx())?;
                let inner = self.ecx().unpack_dyn_trait(dest)?.1;
                trace!("dyn object layout: {:#?}", inner.layout);
                // recurse with the inner type
                return self.visit_field(v, 0, Value::from_mem_place(inner));
            },
            _ => {},
        };

        // If this is a scalar, visit it as such.
        // Things can be aggregates and have scalar layout at the same time, and that
        // is very relevant for `NonNull` and similar structs: We need to visit them
        // at their scalar layout *before* descending into their fields.
        // FIXME: We could avoid some redundant checks here. For newtypes wrapping
        // scalars, we do the same check on every "level" (e.g. first we check
        // MyNewtype and then the scalar in there).
        match v.layout().abi {
            layout::Abi::Uninhabited => {
                self.visit_uninhabited(v)?;
            }
            layout::Abi::Scalar(ref layout) => {
                self.visit_scalar(v, layout)?;
            }
            // FIXME: Should we do something for ScalarPair? Vector?
            _ => {}
        }

        // Check primitive types.  We do this after checking the scalar layout,
        // just to have that done as well.  Primitives can have varying layout,
        // so we check them separately and before aggregate handling.
        // It is CRITICAL that we get this check right, or we might be
        // validating the wrong thing!
        let primitive = match v.layout().fields {
            // Primitives appear as Union with 0 fields -- except for Boxes and fat pointers.
            layout::FieldPlacement::Union(0) => true,
            _ => v.layout().ty.builtin_deref(true).is_some(),
        };
        if primitive {
            return self.visit_primitive(v);
        }

        // Proceed into the fields.
        match v.layout().fields {
            layout::FieldPlacement::Union(fields) => {
                // Empty unions are not accepted by rustc. That's great, it means we can
                // use that as an unambiguous signal for detecting primitives.  Make sure
                // we did not miss any primitive.
                debug_assert!(fields > 0);
                self.visit_union(v)?;
            },
            layout::FieldPlacement::Arbitrary { ref offsets, .. } => {
                for i in 0..offsets.len() {
                    let val = v.project_field(self.ecx(), i as u64)?;
                    self.visit_field(v, i, val)?;
                }
            },
            layout::FieldPlacement::Array { .. } => {
                self.visit_array(v)?;
            }
        }
        Ok(())
    }
}
