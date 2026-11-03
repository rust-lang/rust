// Representing terms
//
// Terms are structured as a straightforward tree. Rather than rely on
// GC, we allocate terms out of a bounded arena (the lifetime of this
// arena is the lifetime 'a that is threaded around).
//
// We assign a unique index to each type/region parameter whose variance
// is to be inferred. We refer to such variables as "inferreds". An
// `InferredIndex` is a newtype'd int representing the index of such
// a variable.

use std::fmt;

use rustc_arena::DroplessArena;
use rustc_attr_ir::{RustcVarianceKind, find_attr};
use rustc_hir::def::DefKind;
use rustc_hir::def_id::{LocalDefId, LocalDefIdMap};
use rustc_middle::ty::{self, TyCtxt};
use tracing::debug;

use self::VarianceTerm::*;

pub(crate) type VarianceTermPtr<'a> = &'a VarianceTerm<'a>;

#[derive(Copy, Clone, Debug)]
pub(crate) struct InferredIndex(pub usize);

#[derive(Copy, Clone)]
pub(crate) enum VarianceTerm<'a> {
    ConstantTerm(ty::Variance),
    TransformTerm(VarianceTermPtr<'a>, VarianceTermPtr<'a>),
    InferredTerm(InferredIndex),
}

impl<'a> fmt::Debug for VarianceTerm<'a> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match *self {
            ConstantTerm(c1) => write!(f, "{c1:?}"),
            TransformTerm(v1, v2) => write!(f, "({v1:?} \u{00D7} {v2:?})"),
            InferredTerm(id) => write!(f, "[{}]", {
                let InferredIndex(i) = id;
                i
            }),
        }
    }
}

// The first pass over the crate simply builds up the set of inferreds.

pub(crate) struct TermsContext<'a, 'tcx> {
    pub tcx: TyCtxt<'tcx>,
    pub arena: &'a DroplessArena,

    /// Explicit variance overrides from attributes.
    pub explicit_variances: Vec<(InferredIndex, ty::Variance)>,

    /// Maps from the node id of an item to the first inferred index
    /// used for its type & region parameters.
    pub inferred_starts: LocalDefIdMap<InferredIndex>,

    /// Maps from an InferredIndex to the term for that variable.
    pub inferred_terms: Vec<VarianceTermPtr<'a>>,
}

pub(crate) fn determine_parameters_to_be_inferred<'a, 'tcx>(
    tcx: TyCtxt<'tcx>,
    arena: &'a DroplessArena,
) -> TermsContext<'a, 'tcx> {
    let mut terms_cx = TermsContext {
        tcx,
        arena,
        inferred_starts: Default::default(),
        inferred_terms: vec![],
        explicit_variances: vec![],
    };

    // See the following for a discussion on dep-graph management.
    //
    // - https://rustc-dev-guide.rust-lang.org/query.html
    // - https://rustc-dev-guide.rust-lang.org/variance.html
    let crate_items = tcx.hir_crate_items(());

    for def_id in crate_items.definitions() {
        debug!("add_inferreds for item {:?}", def_id);

        let def_kind = tcx.def_kind(def_id);

        match def_kind {
            DefKind::Struct | DefKind::Union | DefKind::Enum => {
                terms_cx.add_inferreds_for_item(def_id);

                let adt = tcx.adt_def(def_id);
                for variant in adt.variants() {
                    if let Some(ctor_def_id) = variant.ctor_def_id() {
                        terms_cx.add_inferreds_for_item(ctor_def_id.expect_local());
                    }
                }
            }
            DefKind::Fn | DefKind::AssocFn => terms_cx.add_inferreds_for_item(def_id),
            _ => {}
        }
    }

    terms_cx
}

impl<'a, 'tcx> TermsContext<'a, 'tcx> {
    fn add_inferreds_for_item(&mut self, def_id: LocalDefId) {
        let tcx = self.tcx;
        let generics = tcx.generics_of(def_id);

        if generics.count() == 0 {
            return;
        }

        // Record the start of this item's inferreds.
        let start = self.inferred_terms.len();
        let newly_added = self.inferred_starts.insert(def_id, InferredIndex(start)).is_none();
        assert!(newly_added);

        // Record any parameters with overridden variance.
        for param in &generics.own_params {
            if let Some(variance) =
                find_attr!(tcx, param.def_id, RustcForceVariance(variance) => *variance)
            {
                let variance = match variance {
                    RustcVarianceKind::Covariant => ty::Covariant,
                    RustcVarianceKind::Contravariant => ty::Contravariant,
                    RustcVarianceKind::Invariant => ty::Invariant,
                    RustcVarianceKind::Bivariant => ty::Bivariant,
                };
                self.explicit_variances
                    .push((InferredIndex(start + param.index as usize), variance));
            }
        }

        // N.B., in the code below for writing the results back into the
        // `CrateVariancesMap`, we rely on the fact that all inferreds
        // for a particular item are assigned continuous indices.

        let arena = self.arena;
        self.inferred_terms.extend(
            (start..(start + generics.count()))
                .map(|i| &*arena.alloc(InferredTerm(InferredIndex(i)))),
        );
    }
}
