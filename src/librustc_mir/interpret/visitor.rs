//! Visitor for a run-time value with a given layout: Traverse enums, structs and other compound
//! types until we arrive at the leaves, with custom handling for primitive types.

use std::fmt;

use rustc::ty::layout::{self, TyLayout};
use rustc::ty;
use rustc::mir::interpret::{
    EvalResult,
};

use super::{
    Machine, EvalContext,
};

// How to traverse a value and what to do when we are at the leaves.
// In the future, we might want to turn this into two traits, but so far the
// only implementations we have couldn't share any code anyway.
pub trait ValueVisitor<'a, 'mir, 'tcx, M: Machine<'a, 'mir, 'tcx>>: fmt::Debug {
    // Get this value's layout.
    fn layout(&self) -> TyLayout<'tcx>;

    // Downcast functions.  These change the value as a side-effect.
    fn downcast_enum(&mut self, ectx: &EvalContext<'a, 'mir, 'tcx, M>)
        -> EvalResult<'tcx>;
    fn downcast_dyn_trait(&mut self, ectx: &EvalContext<'a, 'mir, 'tcx, M>)
        -> EvalResult<'tcx>;

    // Visit all fields of a compound.
    // Just call `visit_value` if you want to go on recursively.
    fn visit_fields(&mut self, ectx: &mut EvalContext<'a, 'mir, 'tcx, M>, num_fields: usize)
        -> EvalResult<'tcx>;
    // Optimized handling for arrays -- avoid computing the layout for every field.
    // Also it is the value's responsibility to figure out the length.
    fn visit_array(&mut self, ectx: &mut EvalContext<'a, 'mir, 'tcx, M>) -> EvalResult<'tcx>;
    // Special handling for strings.
    fn visit_str(&mut self, ectx: &mut EvalContext<'a, 'mir, 'tcx, M>)
        -> EvalResult<'tcx>;

    // Actions on the leaves.
    fn visit_scalar(&mut self, ectx: &mut EvalContext<'a, 'mir, 'tcx, M>, layout: &layout::Scalar)
        -> EvalResult<'tcx>;
    fn visit_primitive(&mut self, ectx: &mut EvalContext<'a, 'mir, 'tcx, M>)
        -> EvalResult<'tcx>;
}

impl<'a, 'mir, 'tcx, M: Machine<'a, 'mir, 'tcx>> EvalContext<'a, 'mir, 'tcx, M> {
    pub fn visit_value<V: ValueVisitor<'a, 'mir, 'tcx, M>>(&mut self, v: &mut V) -> EvalResult<'tcx> {
        trace!("visit_value: {:?}", v);

        // If this is a multi-variant layout, we have find the right one and proceed with that.
        // (No benefit from making this recursion, but it is equivalent to that.)
        match v.layout().variants {
            layout::Variants::NicheFilling { .. } |
            layout::Variants::Tagged { .. } => {
                v.downcast_enum(self)?;
                trace!("variant layout: {:#?}", v.layout());
            }
            layout::Variants::Single { .. } => {}
        }

        // Even for single variants, we might be able to get a more refined type:
        // If it is a trait object, switch to the actual type that was used to create it.
        match v.layout().ty.sty {
            ty::Dynamic(..) => {
                v.downcast_dyn_trait(self)?;
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
                Ok(())
            },
            layout::FieldPlacement::Arbitrary { ref offsets, .. } => {
                v.visit_fields(self, offsets.len())
            },
            layout::FieldPlacement::Array { .. } => {
                match v.layout().ty.sty {
                    // Strings have properties that cannot be expressed pointwise.
                    ty::Str => v.visit_str(self),
                    // General case -- might also be SIMD vector or so
                    _ => v.visit_array(self),
                }
            }
        }
    }
}
