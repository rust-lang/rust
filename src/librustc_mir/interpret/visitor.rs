//! Visitor for a run-time value with a given layout: Traverse enums, structs and other compound
//! types until we arrive at the leaves, with custom handling for primitive types.

use std::fmt;

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
pub trait ValueVisitor<'a, 'mir, 'tcx, M: Machine<'a, 'mir, 'tcx>>: fmt::Debug + Sized {
    type V: Value<'a, 'mir, 'tcx, M>;

    // There's a value in here.
    fn value(&self) -> &Self::V;

    // The value's layout (not meant to be overwritten).
    #[inline(always)]
    fn layout(&self) -> TyLayout<'tcx> {
        self.value().layout()
    }

    // Recursie actions, ready to be overloaded.
    /// Visit the current value, dispatching as appropriate to more speicalized visitors.
    #[inline]
    fn visit_value(&mut self, ectx: &mut EvalContext<'a, 'mir, 'tcx, M>)
        -> EvalResult<'tcx>
    {
        self.walk_value(ectx)
    }
    /// Visit the current value as an array.
    #[inline]
    fn visit_array(&mut self, ectx: &mut EvalContext<'a, 'mir, 'tcx, M>)
        -> EvalResult<'tcx>
    {
        self.walk_array(ectx)
    }
    /// Called each time we recurse down to a field of the value, to (a) let
    /// the visitor change its internal state (recording the new current value),
    /// and (b) let the visitor track the "stack" of fields that we descended below.
    fn visit_field(
        &mut self,
        ectx: &mut EvalContext<'a, 'mir, 'tcx, M>,
        val: Self::V,
        field: usize,
    ) -> EvalResult<'tcx>;

    // Actions on the leaves, ready to be overloaded.
    #[inline]
    fn visit_uninhabited(&mut self, _ectx: &mut EvalContext<'a, 'mir, 'tcx, M>)
        -> EvalResult<'tcx>
    { Ok(()) }
    #[inline]
    fn visit_scalar(&mut self, _ectx: &mut EvalContext<'a, 'mir, 'tcx, M>, _layout: &layout::Scalar)
        -> EvalResult<'tcx>
    { Ok(()) }
    #[inline]
    fn visit_primitive(&mut self, _ectx: &mut EvalContext<'a, 'mir, 'tcx, M>)
        -> EvalResult<'tcx>
    { Ok(()) }

    // Default recursors. Not meant to be overloaded.
    fn walk_array(&mut self, ectx: &mut EvalContext<'a, 'mir, 'tcx, M>)
        -> EvalResult<'tcx>
    {
        // Let's get an mplace first.
        let mplace = if self.layout().is_zst() {
            // it's a ZST, the memory content cannot matter
            MPlaceTy::dangling(self.layout(), ectx)
        } else {
            // non-ZST array/slice/str cannot be immediate
            self.value().to_mem_place(ectx)?
        };
        // Now iterate over it.
        for (i, field) in ectx.mplace_array_fields(mplace)?.enumerate() {
            self.visit_field(ectx, Value::from_mem_place(field?), i)?;
        }
        Ok(())
    }
    fn walk_value(&mut self, ectx: &mut EvalContext<'a, 'mir, 'tcx, M>)
        -> EvalResult<'tcx>
    {
        trace!("walk_value: {:?}", self);

        // If this is a multi-variant layout, we have find the right one and proceed with that.
        // (No benefit from making this recursion, but it is equivalent to that.)
        match self.layout().variants {
            layout::Variants::NicheFilling { .. } |
            layout::Variants::Tagged { .. } => {
                let (inner, idx) = self.value().project_downcast(ectx)?;
                trace!("variant layout: {:#?}", inner.layout());
                // recurse with the inner type
                return self.visit_field(ectx, inner, idx);
            }
            layout::Variants::Single { .. } => {}
        }

        // Even for single variants, we might be able to get a more refined type:
        // If it is a trait object, switch to the actual type that was used to create it.
        match self.layout().ty.sty {
            ty::Dynamic(..) => {
                // immediate trait objects are not a thing
                let dest = self.value().to_mem_place(ectx)?;
                let inner = ectx.unpack_dyn_trait(dest)?.1;
                trace!("dyn object layout: {:#?}", inner.layout);
                // recurse with the inner type
                return self.visit_field(ectx, Value::from_mem_place(inner), 0);
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
        match self.layout().abi {
            layout::Abi::Uninhabited => {
                self.visit_uninhabited(ectx)?;
            }
            layout::Abi::Scalar(ref layout) => {
                self.visit_scalar(ectx, layout)?;
            }
            // FIXME: Should we do something for ScalarPair? Vector?
            _ => {}
        }

        // Check primitive types.  We do this after checking the scalar layout,
        // just to have that done as well.  Primitives can have varying layout,
        // so we check them separately and before aggregate handling.
        // It is CRITICAL that we get this check right, or we might be
        // validating the wrong thing!
        let primitive = match self.layout().fields {
            // Primitives appear as Union with 0 fields -- except for Boxes and fat pointers.
            layout::FieldPlacement::Union(0) => true,
            _ => self.layout().ty.builtin_deref(true).is_some(),
        };
        if primitive {
            return self.visit_primitive(ectx);
        }

        // Proceed into the fields.
        match self.layout().fields {
            layout::FieldPlacement::Union(fields) => {
                // Empty unions are not accepted by rustc. That's great, it means we can
                // use that as an unambiguous signal for detecting primitives.  Make sure
                // we did not miss any primitive.
                debug_assert!(fields > 0);
                // We can't traverse unions, their bits are allowed to be anything.
                // The fields don't need to correspond to any bit pattern of the union's fields.
                // See https://github.com/rust-lang/rust/issues/32836#issuecomment-406875389
            },
            layout::FieldPlacement::Arbitrary { ref offsets, .. } => {
                for i in 0..offsets.len() {
                    let val = self.value().project_field(ectx, i as u64)?;
                    self.visit_field(ectx, val, i)?;
                }
            },
            layout::FieldPlacement::Array { .. } => {
                self.visit_array(ectx)?;
            }
        }
        Ok(())
    }
}
