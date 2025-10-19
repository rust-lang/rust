//! Deduces supplementary parameter attributes from MIR.
//!
//! Deduced parameter attributes are those that can only be soundly determined by examining the
//! body of the function instead of just the signature. These can be useful for optimization
//! purposes on a best-effort basis. We compute them here and store them into the crate metadata so
//! dependent crates can use them.
//!
//! Note that this *crucially* relies on codegen *not* doing any more MIR-level transformations
//! after `optimized_mir`! We check for things that are *not* guaranteed to be preserved by MIR
//! transforms, such as which local variables happen to be mutated.

use rustc_hir::def_id::LocalDefId;
use rustc_index::IndexVec;
use rustc_middle::middle::deduced_param_attrs::{DeducedParamAttrs, DeducedReadOnlyParam};
use rustc_middle::mir::visit::{MutatingUseContext, NonMutatingUseContext, PlaceContext, Visitor};
use rustc_middle::mir::*;
use rustc_middle::ty::{self, Ty, TyCtxt};
use rustc_session::config::OptLevel;

/// A visitor that determines which arguments have been mutated. We can't use the mutability field
/// on LocalDecl for this because it has no meaning post-optimization.
struct DeduceReadOnly {
    /// Each bit is indexed by argument number, starting at zero (so 0 corresponds to local decl
    /// 1). The bit is false if the argument may have been mutated or true if we know it hasn't
    /// been up to the point we're at.
    read_only: IndexVec<usize, DeducedReadOnlyParam>,
}

impl DeduceReadOnly {
    /// Returns a new DeduceReadOnly instance.
    fn new(arg_count: usize) -> Self {
        Self { read_only: IndexVec::from_elem_n(DeducedReadOnlyParam::empty(), arg_count) }
    }

    /// Returns whether the given local is a parameter and its index.
    fn as_param(&self, local: Local) -> Option<usize> {
        // Locals and parameters are shifted by `RETURN_PLACE`.
        let param_index = local.as_usize().checked_sub(1)?;
        if param_index < self.read_only.len() { Some(param_index) } else { None }
    }
}

impl<'tcx> Visitor<'tcx> for DeduceReadOnly {
    fn visit_place(&mut self, place: &Place<'tcx>, context: PlaceContext, _location: Location) {
        // We're only interested in arguments.
        let Some(param_index) = self.as_param(place.local) else { return };

        match context {
            // Not mutating, so it's fine.
            PlaceContext::NonUse(..) => {}
            // Dereference is not a mutation.
            _ if place.is_indirect_first_projection() => {}
            // This is a `Drop`. It could disappear at monomorphization, so mark it specially.
            PlaceContext::MutatingUse(MutatingUseContext::Drop)
                // Projection changes the place's type, so `needs_drop(local.ty)` is not
                // `needs_drop(place.ty)`.
                if place.projection.is_empty() => {
                self.read_only[param_index] |= DeducedReadOnlyParam::IF_NO_DROP;
            }
            // This is a mutation, so mark it as such.
            PlaceContext::MutatingUse(..)
            // Whether mutating though a `&raw const` is allowed is still undecided, so we
            // disable any sketchy `readonly` optimizations for now.
            | PlaceContext::NonMutatingUse(NonMutatingUseContext::RawBorrow) => {
                self.read_only[param_index] |= DeducedReadOnlyParam::MUTATED;
            }
            // Not mutating if the parameter is `Freeze`.
            PlaceContext::NonMutatingUse(NonMutatingUseContext::SharedBorrow) => {
                self.read_only[param_index] |= DeducedReadOnlyParam::IF_FREEZE;
            }
            // Not mutating, so it's fine.
            PlaceContext::NonMutatingUse(..) => {}
        }
    }

    fn visit_terminator(&mut self, terminator: &Terminator<'tcx>, location: Location) {
        // OK, this is subtle. Suppose that we're trying to deduce whether `x` in `f` is read-only
        // and we have the following:
        //
        //     fn f(x: BigStruct) { g(x) }
        //     fn g(mut y: BigStruct) { y.foo = 1 }
        //
        // If, at the generated MIR level, `f` turned into something like:
        //
        //      fn f(_1: BigStruct) -> () {
        //          let mut _0: ();
        //          bb0: {
        //              _0 = g(move _1) -> bb1;
        //          }
        //          ...
        //      }
        //
        // then it would be incorrect to mark `x` (i.e. `_1`) as `readonly`, because `g`'s write to
        // its copy of the indirect parameter would actually be a write directly to the pointer that
        // `f` passes. Note that function arguments are the only situation in which this problem can
        // arise: every other use of `move` in MIR doesn't actually write to the value it moves
        // from.
        if let TerminatorKind::Call { ref args, .. } = terminator.kind {
            for arg in args {
                if let Operand::Move(place) = arg.node
                    // We're only interested in arguments.
                    && let Some(param_index) = self.as_param(place.local)
                    && !place.is_indirect_first_projection()
                {
                    self.read_only[param_index] |= DeducedReadOnlyParam::MUTATED;
                }
            }
        };

        self.super_terminator(terminator, location);
    }
}

/// Returns true if values of a given type will never be passed indirectly, regardless of ABI.
fn type_will_always_be_passed_directly(ty: Ty<'_>) -> bool {
    matches!(
        ty.kind(),
        ty::Bool
            | ty::Char
            | ty::Float(..)
            | ty::Int(..)
            | ty::RawPtr(..)
            | ty::Ref(..)
            | ty::Slice(..)
            | ty::Uint(..)
    )
}

/// Returns the deduced parameter attributes for a function.
///
/// Deduced parameter attributes are those that can only be soundly determined by examining the
/// body of the function instead of just the signature. These can be useful for optimization
/// purposes on a best-effort basis. We compute them here and store them into the crate metadata so
/// dependent crates can use them.
#[tracing::instrument(level = "trace", skip(tcx), ret)]
pub(super) fn deduced_param_attrs<'tcx>(
    tcx: TyCtxt<'tcx>,
    def_id: LocalDefId,
) -> &'tcx [DeducedParamAttrs] {
    // This computation is unfortunately rather expensive, so don't do it unless we're optimizing.
    // Also skip it in incremental mode.
    if tcx.sess.opts.optimize == OptLevel::No || tcx.sess.opts.incremental.is_some() {
        return &[];
    }

    // If the Freeze lang item isn't present, then don't bother.
    if tcx.lang_items().freeze_trait().is_none() {
        return &[];
    }

    // Codegen won't use this information for anything if all the function parameters are passed
    // directly. Detect that and bail, for compilation speed.
    let fn_ty = tcx.type_of(def_id).instantiate_identity();
    if matches!(fn_ty.kind(), ty::FnDef(..))
        && fn_ty
            .fn_sig(tcx)
            .inputs()
            .skip_binder()
            .iter()
            .cloned()
            .all(type_will_always_be_passed_directly)
    {
        return &[];
    }

    // Don't deduce any attributes for functions that have no MIR.
    if !tcx.is_mir_available(def_id) {
        return &[];
    }

    // Grab the optimized MIR. Analyze it to determine which arguments have been mutated.
    let body: &Body<'tcx> = tcx.optimized_mir(def_id);
    let mut deduce_read_only = DeduceReadOnly::new(body.arg_count);
    deduce_read_only.visit_body(body);
    tracing::trace!(?deduce_read_only.read_only);

    let mut deduced_param_attrs: &[_] = tcx.arena.alloc_from_iter(
        deduce_read_only.read_only.into_iter().map(|read_only| DeducedParamAttrs { read_only }),
    );

    // Trailing parameters past the size of the `deduced_param_attrs` array are assumed to have the
    // default set of attributes, so we don't have to store them explicitly. Pop them off to save a
    // few bytes in metadata.
    while let Some((last, rest)) = deduced_param_attrs.split_last()
        && last.is_default()
    {
        deduced_param_attrs = rest;
    }

    deduced_param_attrs
}
