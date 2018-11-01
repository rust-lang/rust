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
    fn force_allocation(
        self,
        ectx: &mut EvalContext<'a, 'mir, 'tcx, M>,
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
    fn force_allocation(
        self,
        _ectx: &mut EvalContext<'a, 'mir, 'tcx, M>,
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
    fn force_allocation(
        self,
        _ectx: &mut EvalContext<'a, 'mir, 'tcx, M>,
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
    fn force_allocation(
        self,
        ectx: &mut EvalContext<'a, 'mir, 'tcx, M>,
    ) -> EvalResult<'tcx, MPlaceTy<'tcx, M::PointerTag>> {
        ectx.force_allocation(self)
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

    // Replace the value by `val`, which must be the `field`th field of `self`, then call
    // `visit_value` and then un-do everything that might have happened to the visitor state.
    // The point of this is that some visitors keep a stack of fields that we projected below,
    // and this lets us avoid copying that stack; instead they will pop the stack after
    // executing `visit_value`.
    fn visit_field(
        &mut self,
        ectx: &mut EvalContext<'a, 'mir, 'tcx, M>,
        val: Self::V,
        field: usize,
    ) -> EvalResult<'tcx>;

    // A chance for the visitor to do special (different or more efficient) handling for some
    // array types.  Return `true` if the value was handled and we should return.
    #[inline]
    fn handle_array(&mut self, _ectx: &EvalContext<'a, 'mir, 'tcx, M>)
        -> EvalResult<'tcx, bool>
    {
        Ok(false)
    }

    // Execute visitor on the current value.  Used for recursing.
    #[inline]
    fn visit(&mut self, ectx: &mut EvalContext<'a, 'mir, 'tcx, M>)
        -> EvalResult<'tcx>
    {
        ectx.walk_value(self)
    }

    // Actions on the leaves.
    fn visit_uninhabited(&mut self, _ectx: &mut EvalContext<'a, 'mir, 'tcx, M>)
        -> EvalResult<'tcx>
    { Ok(()) }
    fn visit_scalar(&mut self, _ectx: &mut EvalContext<'a, 'mir, 'tcx, M>, _layout: &layout::Scalar)
        -> EvalResult<'tcx>
    { Ok(()) }
    fn visit_primitive(&mut self, _ectx: &mut EvalContext<'a, 'mir, 'tcx, M>)
        -> EvalResult<'tcx>
    { Ok(()) }
}

impl<'a, 'mir, 'tcx, M: Machine<'a, 'mir, 'tcx>> EvalContext<'a, 'mir, 'tcx, M> {
    pub fn walk_value<V: ValueVisitor<'a, 'mir, 'tcx, M>>(
        &mut self,
        v: &mut V,
    ) -> EvalResult<'tcx> {
        trace!("walk_value: {:?}", v);

        // If this is a multi-variant layout, we have find the right one and proceed with that.
        // (No benefit from making this recursion, but it is equivalent to that.)
        match v.layout().variants {
            layout::Variants::NicheFilling { .. } |
            layout::Variants::Tagged { .. } => {
                let (inner, idx) = v.value().project_downcast(self)?;
                trace!("variant layout: {:#?}", inner.layout());
                // recurse with the inner type
                return v.visit_field(self, inner, idx);
            }
            layout::Variants::Single { .. } => {}
        }

        // Even for single variants, we might be able to get a more refined type:
        // If it is a trait object, switch to the actual type that was used to create it.
        match v.layout().ty.sty {
            ty::Dynamic(..) => {
                // immediate trait objects are not a thing
                let dest = v.value().force_allocation(self)?;
                let inner = self.unpack_dyn_trait(dest)?.1;
                trace!("dyn object layout: {:#?}", inner.layout);
                // recurse with the inner type
                return v.visit_field(self, Value::from_mem_place(inner), 0);
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
                v.visit_uninhabited(self)?;
            }
            layout::Abi::Scalar(ref layout) => {
                v.visit_scalar(self, layout)?;
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
            return v.visit_primitive(self);
        }

        // Proceed into the fields.
        match v.layout().fields {
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
                    let val = v.value().project_field(self, i as u64)?;
                    v.visit_field(self, val, i)?;
                }
            },
            layout::FieldPlacement::Array { .. } => {
                if !v.handle_array(self)? {
                    // We still have to work!
                    // Let's get an mplace first.
                    let mplace = if v.layout().is_zst() {
                        // it's a ZST, the memory content cannot matter
                        MPlaceTy::dangling(v.layout(), self)
                    } else {
                        // non-ZST array/slice/str cannot be immediate
                        v.value().force_allocation(self)?
                    };
                    // Now iterate over it.
                    for (i, field) in self.mplace_array_fields(mplace)?.enumerate() {
                        v.visit_field(self, Value::from_mem_place(field?), i)?;
                    }
                }
            }
        }
        Ok(())
    }
}
