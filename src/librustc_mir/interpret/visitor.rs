//! Visitor for a run-time value with a given layout: Traverse enums, structs and other compound
//! types until we arrive at the leaves, with custom handling for primitive types.

use rustc::ty::layout::{self, TyLayout};
use rustc::ty;
use rustc::mir::interpret::{
    EvalResult,
};

use super::{
    Machine, EvalContext, MPlaceTy, PlaceTy, OpTy, ImmTy,
};

// A thing that we can project into, and that has a layout.
// This wouldn't have to depend on `Machine` but with the current type inference,
// that's just more convenient to work with (avoids repeating all the `Machine` bounds).
pub trait Value<'a, 'mir, 'tcx, M: Machine<'a, 'mir, 'tcx>>: Copy
{
    /// Get this value's layout.
    fn layout(&self) -> TyLayout<'tcx>;

    /// Make this into an `OpTy`.
    fn to_op(
        self,
        ecx: &EvalContext<'a, 'mir, 'tcx, M>,
    ) -> EvalResult<'tcx, OpTy<'tcx, M::PointerTag>>;

    /// Create this from an `MPlaceTy`.
    fn from_mem_place(MPlaceTy<'tcx, M::PointerTag>) -> Self;

    /// Project to the given enum variant.
    fn project_downcast(
        self,
        ecx: &EvalContext<'a, 'mir, 'tcx, M>,
        variant: usize,
    ) -> EvalResult<'tcx, Self>;

    /// Project to the n-th field.
    fn project_field(
        self,
        ecx: &mut EvalContext<'a, 'mir, 'tcx, M>,
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
    fn to_op(
        self,
        _ecx: &EvalContext<'a, 'mir, 'tcx, M>,
    ) -> EvalResult<'tcx, OpTy<'tcx, M::PointerTag>> {
        Ok(self)
    }

    #[inline(always)]
    fn from_mem_place(mplace: MPlaceTy<'tcx, M::PointerTag>) -> Self {
        mplace.into()
    }

    #[inline(always)]
    fn project_downcast(
        self,
        ecx: &EvalContext<'a, 'mir, 'tcx, M>,
        variant: usize,
    ) -> EvalResult<'tcx, Self> {
        ecx.operand_downcast(self, variant)
    }

    #[inline(always)]
    fn project_field(
        self,
        ecx: &mut EvalContext<'a, 'mir, 'tcx, M>,
        field: u64,
    ) -> EvalResult<'tcx, Self> {
        ecx.operand_field(self, field)
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
    fn to_op(
        self,
        _ecx: &EvalContext<'a, 'mir, 'tcx, M>,
    ) -> EvalResult<'tcx, OpTy<'tcx, M::PointerTag>> {
        Ok(self.into())
    }

    #[inline(always)]
    fn from_mem_place(mplace: MPlaceTy<'tcx, M::PointerTag>) -> Self {
        mplace
    }

    #[inline(always)]
    fn project_downcast(
        self,
        ecx: &EvalContext<'a, 'mir, 'tcx, M>,
        variant: usize,
    ) -> EvalResult<'tcx, Self> {
        ecx.mplace_downcast(self, variant)
    }

    #[inline(always)]
    fn project_field(
        self,
        ecx: &mut EvalContext<'a, 'mir, 'tcx, M>,
        field: u64,
    ) -> EvalResult<'tcx, Self> {
        ecx.mplace_field(self, field)
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
    fn to_op(
        self,
        ecx: &EvalContext<'a, 'mir, 'tcx, M>,
    ) -> EvalResult<'tcx, OpTy<'tcx, M::PointerTag>> {
        ecx.place_to_op(self)
    }

    #[inline(always)]
    fn from_mem_place(mplace: MPlaceTy<'tcx, M::PointerTag>) -> Self {
        mplace.into()
    }

    #[inline(always)]
    fn project_downcast(
        self,
        ecx: &EvalContext<'a, 'mir, 'tcx, M>,
        variant: usize,
    ) -> EvalResult<'tcx, Self> {
        ecx.place_downcast(self, variant)
    }

    #[inline(always)]
    fn project_field(
        self,
        ecx: &mut EvalContext<'a, 'mir, 'tcx, M>,
        field: u64,
    ) -> EvalResult<'tcx, Self> {
        ecx.place_field(self, field)
    }
}

// How to traverse a value and what to do when we are at the leaves.
pub trait ValueVisitor<'a, 'mir, 'tcx: 'mir+'a, M: Machine<'a, 'mir, 'tcx>>: Sized {
    type V: Value<'a, 'mir, 'tcx, M>;

    /// The visitor must have an `EvalContext` in it.
    fn ecx(&mut self) -> &mut EvalContext<'a, 'mir, 'tcx, M>;

    // Recursive actions, ready to be overloaded.
    /// Visit the given value, dispatching as appropriate to more specialized visitors.
    #[inline(always)]
    fn visit_value(&mut self, v: Self::V) -> EvalResult<'tcx>
    {
        self.walk_value(v)
    }
    /// Visit the given value as a union.  No automatic recursion can happen here.
    #[inline(always)]
    fn visit_union(&mut self, _v: Self::V) -> EvalResult<'tcx>
    {
        Ok(())
    }
    /// Visit this vale as an aggregate, you are even getting an iterator yielding
    /// all the fields (still in an `EvalResult`, you have to do error handling yourself).
    /// Recurses into the fields.
    #[inline(always)]
    fn visit_aggregate(
        &mut self,
        v: Self::V,
        fields: impl Iterator<Item=EvalResult<'tcx, Self::V>>,
    ) -> EvalResult<'tcx> {
        self.walk_aggregate(v, fields)
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

    /// Called whenever we reach a value with uninhabited layout.
    /// Recursing to fields will *always* continue after this!  This is not meant to control
    /// whether and how we descend recursively/ into the scalar's fields if there are any, it is
    /// meant to provide the chance for additional checks when a value of uninhabited layout is
    /// detected.
    #[inline(always)]
    fn visit_uninhabited(&mut self) -> EvalResult<'tcx>
    { Ok(()) }
    /// Called whenever we reach a value with scalar layout.
    /// We do NOT provide a `ScalarMaybeUndef` here to avoid accessing memory if the visitor is not
    /// even interested in scalars.
    /// Recursing to fields will *always* continue after this!  This is not meant to control
    /// whether and how we descend recursively/ into the scalar's fields if there are any, it is
    /// meant to provide the chance for additional checks when a value of scalar layout is detected.
    #[inline(always)]
    fn visit_scalar(&mut self, _v: Self::V, _layout: &layout::Scalar) -> EvalResult<'tcx>
    { Ok(()) }

    /// Called whenever we reach a value of primitive type.  There can be no recursion
    /// below such a value.  This is the leave function.
    #[inline(always)]
    fn visit_primitive(&mut self, _val: ImmTy<'tcx, M::PointerTag>) -> EvalResult<'tcx>
    { Ok(()) }

    // Default recursors. Not meant to be overloaded.
    fn walk_aggregate(
        &mut self,
        v: Self::V,
        fields: impl Iterator<Item=EvalResult<'tcx, Self::V>>,
    ) -> EvalResult<'tcx> {
        // Now iterate over it.
        for (idx, field_val) in fields.enumerate() {
            self.visit_field(v, idx, field_val?)?;
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
                let op = v.to_op(self.ecx())?;
                let idx = self.ecx().read_discriminant(op)?.1;
                let inner = v.project_downcast(self.ecx(), idx)?;
                trace!("walk_value: variant layout: {:#?}", inner.layout());
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
                let dest = v.to_op(self.ecx())?.to_mem_place();
                let inner = self.ecx().unpack_dyn_trait(dest)?.1;
                trace!("walk_value: dyn object layout: {:#?}", inner.layout);
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
                self.visit_uninhabited()?;
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
            let op = v.to_op(self.ecx())?;
            let val = self.ecx().read_immediate(op)?;
            return self.visit_primitive(val);
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
                // We collect in a vec because otherwise there are lifetime errors:
                // Projecting to a field needs (mutable!) access to `ecx`.
                let fields: Vec<EvalResult<'tcx, Self::V>> =
                    (0..offsets.len()).map(|i| {
                        v.project_field(self.ecx(), i as u64)
                    })
                    .collect();
                self.visit_aggregate(v, fields.into_iter())?;
            },
            layout::FieldPlacement::Array { .. } => {
                // Let's get an mplace first.
                let mplace = if v.layout().is_zst() {
                    // it's a ZST, the memory content cannot matter
                    MPlaceTy::dangling(v.layout(), self.ecx())
                } else {
                    // non-ZST array/slice/str cannot be immediate
                    v.to_op(self.ecx())?.to_mem_place()
                };
                // Now we can go over all the fields.
                let iter = self.ecx().mplace_array_fields(mplace)?
                    .map(|f| f.and_then(|f| {
                        Ok(Value::from_mem_place(f))
                    }));
                self.visit_aggregate(v, iter)?;
            }
        }
        Ok(())
    }
}
