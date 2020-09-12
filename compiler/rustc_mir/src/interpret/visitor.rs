//! Visitor for a run-time value with a given layout: Traverse enums, structs and other compound
//! types until we arrive at the leaves, with custom handling for primitive types.

use rustc_middle::mir::interpret::InterpResult;
use rustc_middle::ty;
use rustc_middle::ty::layout::TyAndLayout;
use rustc_target::abi::{FieldsShape, VariantIdx, Variants};

use std::num::NonZeroUsize;

use super::{InterpCx, MPlaceTy, Machine, OpTy};

// A thing that we can project into, and that has a layout.
// This wouldn't have to depend on `Machine` but with the current type inference,
// that's just more convenient to work with (avoids repeating all the `Machine` bounds).
pub trait Value<'mir, 'tcx, M: Machine<'mir, 'tcx>>: Copy {
    /// Gets this value's layout.
    fn layout(&self) -> TyAndLayout<'tcx>;

    /// Makes this into an `OpTy`.
    fn to_op(self, ecx: &InterpCx<'mir, 'tcx, M>) -> InterpResult<'tcx, OpTy<'tcx, M::PointerTag>>;

    /// Creates this from an `MPlaceTy`.
    fn from_mem_place(mplace: MPlaceTy<'tcx, M::PointerTag>) -> Self;

    /// Projects to the given enum variant.
    fn project_downcast(
        self,
        ecx: &InterpCx<'mir, 'tcx, M>,
        variant: VariantIdx,
    ) -> InterpResult<'tcx, Self>;

    /// Projects to the n-th field.
    fn project_field(self, ecx: &InterpCx<'mir, 'tcx, M>, field: usize)
    -> InterpResult<'tcx, Self>;
}

// Operands and memory-places are both values.
// Places in general are not due to `place_field` having to do `force_allocation`.
impl<'mir, 'tcx: 'mir, M: Machine<'mir, 'tcx>> Value<'mir, 'tcx, M> for OpTy<'tcx, M::PointerTag> {
    #[inline(always)]
    fn layout(&self) -> TyAndLayout<'tcx> {
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
        field: usize,
    ) -> InterpResult<'tcx, Self> {
        ecx.operand_field(self, field)
    }
}

impl<'mir, 'tcx: 'mir, M: Machine<'mir, 'tcx>> Value<'mir, 'tcx, M>
    for MPlaceTy<'tcx, M::PointerTag>
{
    #[inline(always)]
    fn layout(&self) -> TyAndLayout<'tcx> {
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
        field: usize,
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

            /// `read_discriminant` can be hooked for better error messages.
            #[inline(always)]
            fn read_discriminant(
                &mut self,
                op: OpTy<'tcx, M::PointerTag>,
            ) -> InterpResult<'tcx, VariantIdx> {
                Ok(self.ecx().read_discriminant(op)?.1)
            }

            // Recursive actions, ready to be overloaded.
            /// Visits the given value, dispatching as appropriate to more specialized visitors.
            #[inline(always)]
            fn visit_value(&mut self, v: Self::V) -> InterpResult<'tcx>
            {
                self.walk_value(v)
            }
            /// Visits the given value as a union. No automatic recursion can happen here.
            #[inline(always)]
            fn visit_union(&mut self, _v: Self::V, _fields: NonZeroUsize) -> InterpResult<'tcx>
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
            /// This gives the visitor the chance to track the stack of nested fields that
            /// we are descending through.
            #[inline(always)]
            fn visit_variant(
                &mut self,
                _old_val: Self::V,
                _variant: VariantIdx,
                new_val: Self::V,
            ) -> InterpResult<'tcx> {
                self.visit_value(new_val)
            }

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

                // Special treatment for special types, where the (static) layout is not sufficient.
                match *v.layout().ty.kind() {
                    // If it is a trait object, switch to the real type that was used to create it.
                    ty::Dynamic(..) => {
                        // immediate trait objects are not a thing
                        let dest = v.to_op(self.ecx())?.assert_mem_place(self.ecx());
                        let inner = self.ecx().unpack_dyn_trait(dest)?.1;
                        trace!("walk_value: dyn object layout: {:#?}", inner.layout);
                        // recurse with the inner type
                        return self.visit_field(v, 0, Value::from_mem_place(inner));
                    },
                    // Slices do not need special handling here: they have `Array` field
                    // placement with length 0, so we enter the `Array` case below which
                    // indirectly uses the metadata to determine the actual length.
                    _ => {},
                };

                // Visit the fields of this value.
                match v.layout().fields {
                    FieldsShape::Primitive => {},
                    FieldsShape::Union(fields) => {
                        self.visit_union(v, fields)?;
                    },
                    FieldsShape::Arbitrary { ref offsets, .. } => {
                        // FIXME: We collect in a vec because otherwise there are lifetime
                        // errors: Projecting to a field needs access to `ecx`.
                        let fields: Vec<InterpResult<'tcx, Self::V>> =
                            (0..offsets.len()).map(|i| {
                                v.project_field(self.ecx(), i)
                            })
                            .collect();
                        self.visit_aggregate(v, fields.into_iter())?;
                    },
                    FieldsShape::Array { .. } => {
                        // Let's get an mplace first.
                        let mplace = v.to_op(self.ecx())?.assert_mem_place(self.ecx());
                        // Now we can go over all the fields.
                        // This uses the *run-time length*, i.e., if we are a slice,
                        // the dynamic info from the metadata is used.
                        let iter = self.ecx().mplace_array_fields(mplace)?
                            .map(|f| f.and_then(|f| {
                                Ok(Value::from_mem_place(f))
                            }));
                        self.visit_aggregate(v, iter)?;
                    }
                }

                match v.layout().variants {
                    // If this is a multi-variant layout, find the right variant and proceed
                    // with *its* fields.
                    Variants::Multiple { .. } => {
                        let op = v.to_op(self.ecx())?;
                        let idx = self.read_discriminant(op)?;
                        let inner = v.project_downcast(self.ecx(), idx)?;
                        trace!("walk_value: variant layout: {:#?}", inner.layout());
                        // recurse with the inner type
                        self.visit_variant(v, idx, inner)
                    }
                    // For single-variant layouts, we already did anything there is to do.
                    Variants::Single { .. } => Ok(())
                }
            }
        }
    }
}

make_value_visitor!(ValueVisitor,);
make_value_visitor!(MutValueVisitor, mut);
