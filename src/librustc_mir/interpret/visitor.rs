//! Visitor for a run-time value with a given layout: Traverse enums, structs and other compound
//! types until we arrive at the leaves, with custom handling for primitive types.

use rustc::ty::layout::{self, TyLayout, VariantIdx};
use rustc::ty;
use rustc::mir::interpret::{
    InterpResult,
};

use super::{
    Machine, InterpCx, MPlaceTy, OpTy,
};

// A thing that we can project into, and that has a layout.
// This wouldn't have to depend on `Machine` but with the current type inference,
// that's just more convenient to work with (avoids repeating all the `Machine` bounds).
pub trait Value<'mir, 'tcx, M: Machine<'mir, 'tcx>>: Copy {
    /// Gets this value's layout.
    fn layout(&self) -> TyLayout<'tcx>;

    /// Makes this into an `OpTy`.
    fn to_op(
        self,
        ecx: &InterpCx<'mir, 'tcx, M>,
    ) -> InterpResult<'tcx, OpTy<'tcx, M::PointerTag>>;

    /// Creates this from an `MPlaceTy`.
    fn from_mem_place(mplace: MPlaceTy<'tcx, M::PointerTag>) -> Self;

    /// Projects to the given enum variant.
    fn project_downcast(
        self,
        ecx: &InterpCx<'mir, 'tcx, M>,
        variant: VariantIdx,
    ) -> InterpResult<'tcx, Self>;

    /// Projects to the n-th field.
    fn project_field(
        self,
        ecx: &InterpCx<'mir, 'tcx, M>,
        field: u64,
    ) -> InterpResult<'tcx, Self>;
}

// Operands and memory-places are both values.
// Places in general are not due to `place_field` having to do `force_allocation`.
impl<'mir, 'tcx, M: Machine<'mir, 'tcx>> Value<'mir, 'tcx, M> for OpTy<'tcx, M::PointerTag> {
    #[inline(always)]
    fn layout(&self) -> TyLayout<'tcx> {
        self.layout
    }

    #[inline(always)]
    fn to_op(
        self,
        _ecx: &InterpCx<'mir, 'tcx, M>,
    ) -> InterpResult<'tcx, OpTy<'tcx, M::PointerTag>> {
        Ok(self)
    }

    #[inline(always)]
    fn from_mem_place(mplace: MPlaceTy<'tcx, M::PointerTag>) -> Self {
        mplace.into()
    }

    #[inline(always)]
    fn project_downcast(
        self,
        ecx: &InterpCx<'mir, 'tcx, M>,
        variant: VariantIdx,
    ) -> InterpResult<'tcx, Self> {
        ecx.operand_downcast(self, variant)
    }

    #[inline(always)]
    fn project_field(
        self,
        ecx: &InterpCx<'mir, 'tcx, M>,
        field: u64,
    ) -> InterpResult<'tcx, Self> {
        ecx.operand_field(self, field)
    }
}

impl<'mir, 'tcx, M: Machine<'mir, 'tcx>> Value<'mir, 'tcx, M> for MPlaceTy<'tcx, M::PointerTag> {
    #[inline(always)]
    fn layout(&self) -> TyLayout<'tcx> {
        self.layout
    }

    #[inline(always)]
    fn to_op(
        self,
        _ecx: &InterpCx<'mir, 'tcx, M>,
    ) -> InterpResult<'tcx, OpTy<'tcx, M::PointerTag>> {
        Ok(self.into())
    }

    #[inline(always)]
    fn from_mem_place(mplace: MPlaceTy<'tcx, M::PointerTag>) -> Self {
        mplace
    }

    #[inline(always)]
    fn project_downcast(
        self,
        ecx: &InterpCx<'mir, 'tcx, M>,
        variant: VariantIdx,
    ) -> InterpResult<'tcx, Self> {
        ecx.mplace_downcast(self, variant)
    }

    #[inline(always)]
    fn project_field(
        self,
        ecx: &InterpCx<'mir, 'tcx, M>,
        field: u64,
    ) -> InterpResult<'tcx, Self> {
        ecx.mplace_field(self, field)
    }
}

macro_rules! make_value_visitor {
    ($visitor_trait_name:ident, $($mutability:ident)?) => {
        // How to traverse a value and what to do when we are at the leaves.
        pub trait $visitor_trait_name<'mir, 'tcx: 'mir, M: Machine<'mir, 'tcx>>: Sized {
            type V: Value<'mir, 'tcx, M>;

            /// The visitor must have an `InterpCx` in it.
            fn ecx(&$($mutability)? self)
                -> &$($mutability)? InterpCx<'mir, 'tcx, M>;

            // Recursive actions, ready to be overloaded.
            /// Visits the given value, dispatching as appropriate to more specialized visitors.
            #[inline(always)]
            fn visit_value(&mut self, v: Self::V) -> InterpResult<'tcx>
            {
                self.walk_value(v)
            }
            /// Visits the given value as a union. No automatic recursion can happen here.
            #[inline(always)]
            fn visit_union(&mut self, _v: Self::V) -> InterpResult<'tcx>
            {
                Ok(())
            }
            /// Visits this value as an aggregate, you are getting an iterator yielding
            /// all the fields (still in an `InterpResult`, you have to do error handling yourself).
            /// Recurses into the fields.
            #[inline(always)]
            fn visit_aggregate(
                &mut self,
                v: Self::V,
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
                _old_val: Self::V,
                _field: usize,
                new_val: Self::V,
            ) -> InterpResult<'tcx> {
                self.visit_value(new_val)
            }

            /// Called when recursing into an enum variant.
            #[inline(always)]
            fn visit_variant(
                &mut self,
                _old_val: Self::V,
                _variant: VariantIdx,
                new_val: Self::V,
            ) -> InterpResult<'tcx> {
                self.visit_value(new_val)
            }

            /// Called whenever we reach a value with uninhabited layout.
            /// Recursing to fields will *always* continue after this!  This is not meant to control
            /// whether and how we descend recursively/ into the scalar's fields if there are any,
            /// it is meant to provide the chance for additional checks when a value of uninhabited
            /// layout is detected.
            #[inline(always)]
            fn visit_uninhabited(&mut self) -> InterpResult<'tcx>
            { Ok(()) }
            /// Called whenever we reach a value with scalar layout.
            /// We do NOT provide a `ScalarMaybeUndef` here to avoid accessing memory if the
            /// visitor is not even interested in scalars.
            /// Recursing to fields will *always* continue after this!  This is not meant to control
            /// whether and how we descend recursively/ into the scalar's fields if there are any,
            /// it is meant to provide the chance for additional checks when a value of scalar
            /// layout is detected.
            #[inline(always)]
            fn visit_scalar(&mut self, _v: Self::V, _layout: &layout::Scalar) -> InterpResult<'tcx>
            { Ok(()) }

            /// Called whenever we reach a value of primitive type. There can be no recursion
            /// below such a value. This is the leaf function.
            /// We do *not* provide an `ImmTy` here because some implementations might want
            /// to write to the place this primitive lives in.
            #[inline(always)]
            fn visit_primitive(&mut self, _v: Self::V) -> InterpResult<'tcx>
            { Ok(()) }

            // Default recursors. Not meant to be overloaded.
            fn walk_aggregate(
                &mut self,
                v: Self::V,
                fields: impl Iterator<Item=InterpResult<'tcx, Self::V>>,
            ) -> InterpResult<'tcx> {
                // Now iterate over it.
                for (idx, field_val) in fields.enumerate() {
                    self.visit_field(v, idx, field_val?)?;
                }
                Ok(())
            }
            fn walk_value(&mut self, v: Self::V) -> InterpResult<'tcx>
            {
                trace!("walk_value: type: {}", v.layout().ty);
                // If this is a multi-variant layout, we have to find the right one and proceed with
                // that.
                match v.layout().variants {
                    layout::Variants::Multiple { .. } => {
                        let op = v.to_op(self.ecx())?;
                        let idx = self.ecx().read_discriminant(op)?.1;
                        let inner = v.project_downcast(self.ecx(), idx)?;
                        trace!("walk_value: variant layout: {:#?}", inner.layout());
                        // recurse with the inner type
                        return self.visit_variant(v, idx, inner);
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
                    ty::Generator(..) => {
                        // FIXME: Generator layout is lying: it claims a whole bunch of fields exist
                        // when really many of them can be uninitialized.
                        // Just treat them as a union for now, until hopefully the layout
                        // computation is fixed.
                        return self.visit_union(v);
                    }
                    _ => {},
                };

                // If this is a scalar, visit it as such.
                // Things can be aggregates and have scalar layout at the same time, and that
                // is very relevant for `NonNull` and similar structs: We need to visit them
                // at their scalar layout *before* descending into their fields.
                // FIXME: We could avoid some redundant checks here. For newtypes wrapping
                // scalars, we do the same check on every "level" (e.g., first we check
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
                    // Primitives appear as Union with 0 fields - except for Boxes and fat pointers.
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
                        assert!(fields > 0);
                        self.visit_union(v)
                    },
                    layout::FieldPlacement::Arbitrary { ref offsets, .. } => {
                        // FIXME: We collect in a vec because otherwise there are lifetime
                        // errors: Projecting to a field needs access to `ecx`.
                        let fields: Vec<InterpResult<'tcx, Self::V>> =
                            (0..offsets.len()).map(|i| {
                                v.project_field(self.ecx(), i as u64)
                            })
                            .collect();
                        self.visit_aggregate(v, fields.into_iter())
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
                        self.visit_aggregate(v, iter)
                    }
                }
            }
        }
    }
}

make_value_visitor!(ValueVisitor,);
make_value_visitor!(MutValueVisitor,mut);
