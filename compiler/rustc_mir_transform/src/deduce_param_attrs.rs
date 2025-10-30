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
use rustc_middle::middle::deduced_param_attrs::{DeducedParamAttrs, UsageSummary};
use rustc_middle::mir::visit::{MutatingUseContext, NonMutatingUseContext, PlaceContext, Visitor};
use rustc_middle::mir::*;
use rustc_middle::ty::{self, Ty, TyCtxt};
use rustc_session::config::OptLevel;

/// A visitor that determines how a return place and arguments are used inside MIR body.
/// To determine whether a local is mutated we can't use the mutability field on LocalDecl
/// because it has no meaning post-optimization.
struct DeduceParamAttrs {
    /// Summarizes how a return place and arguments are used inside MIR body.
    usage: IndexVec<Local, UsageSummary>,
}

impl DeduceParamAttrs {
    /// Returns a new DeduceParamAttrs instance.
    fn new(body: &Body<'_>) -> Self {
        let mut this =
            Self { usage: IndexVec::from_elem_n(UsageSummary::empty(), body.arg_count + 1) };
        // Code generation indicates that a return place is writable. To avoid setting both
        // `readonly` and `writable` attributes, when return place is never written to, mark it as
        // mutated.
        this.usage[RETURN_PLACE] |= UsageSummary::MUTATE;
        this
    }

    /// Returns whether a local is the return place or an argument and returns its index.
    fn as_param(&self, local: Local) -> Option<Local> {
        if local.index() < self.usage.len() { Some(local) } else { None }
    }
}

impl<'tcx> Visitor<'tcx> for DeduceParamAttrs {
    fn visit_place(&mut self, place: &Place<'tcx>, context: PlaceContext, _location: Location) {
        // We're only interested in the return place or an argument.
        let Some(i) = self.as_param(place.local) else { return };

        match context {
            // Not actually using the local.
            PlaceContext::NonUse(..) => {}
            // Neither mutated nor captured.
            _ if place.is_indirect_first_projection() => {}
            // This is a `Drop`. It could disappear at monomorphization, so mark it specially.
            PlaceContext::MutatingUse(MutatingUseContext::Drop)
                // Projection changes the place's type, so `needs_drop(local.ty)` is not
                // `needs_drop(place.ty)`.
                if place.projection.is_empty() => {
                    self.usage[i] |= UsageSummary::DROP;
            }
            PlaceContext::MutatingUse(
                  MutatingUseContext::Call
                | MutatingUseContext::Yield
                | MutatingUseContext::Drop
                | MutatingUseContext::Borrow
                | MutatingUseContext::RawBorrow) => {
                self.usage[i] |= UsageSummary::MUTATE;
                self.usage[i] |= UsageSummary::CAPTURE;
            }
            PlaceContext::MutatingUse(
                  MutatingUseContext::Store
                | MutatingUseContext::SetDiscriminant
                | MutatingUseContext::AsmOutput
                | MutatingUseContext::Projection
                | MutatingUseContext::Retag) => {
                self.usage[i] |= UsageSummary::MUTATE;
            }
            | PlaceContext::NonMutatingUse(NonMutatingUseContext::RawBorrow) => {
                // Whether mutating though a `&raw const` is allowed is still undecided, so we
                // disable any sketchy `readonly` optimizations for now.
                self.usage[i] |= UsageSummary::MUTATE;
                self.usage[i] |= UsageSummary::CAPTURE;
            }
            PlaceContext::NonMutatingUse(NonMutatingUseContext::SharedBorrow) => {
                // Not mutating if the parameter is `Freeze`.
                self.usage[i] |= UsageSummary::SHARED_BORROW;
                self.usage[i] |= UsageSummary::CAPTURE;
            }
            // Not mutating, so it's fine.
            PlaceContext::NonMutatingUse(
                  NonMutatingUseContext::Inspect
                | NonMutatingUseContext::Copy
                | NonMutatingUseContext::Move
                | NonMutatingUseContext::FakeBorrow
                | NonMutatingUseContext::PlaceMention
                | NonMutatingUseContext::Projection) => {}
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
                    && !place.is_indirect_first_projection()
                    && let Some(i) = self.as_param(place.local)
                {
                    self.usage[i] |= UsageSummary::MUTATE;
                    self.usage[i] |= UsageSummary::CAPTURE;
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
            .inputs_and_output()
            .skip_binder()
            .iter()
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
    // Arguments spread at ABI level are currently unsupported.
    if body.spread_arg.is_some() {
        return &[];
    }

    let mut deduce = DeduceParamAttrs::new(body);
    deduce.visit_body(body);
    tracing::trace!(?deduce.usage);

    let mut deduced_param_attrs: &[_] = tcx
        .arena
        .alloc_from_iter(deduce.usage.into_iter().map(|usage| DeducedParamAttrs { usage }));

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
