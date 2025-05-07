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

/// The first pass over the crate simply builds up the set of inferreds.

pub(crate) struct TermsContext<'a, 'tcx> {
    pub tcx: TyCtxt<'tcx>,
    pub arena: &'a DroplessArena,

    /// For marker types, `UnsafeCell`, and other lang items where
    /// variance is hardcoded, records the item-id and the hardcoded
    /// variance.
    pub lang_items: Vec<(LocalDefId, Vec<ty::Variance>)>,

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

        lang_items: lang_items(tcx),
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
            DefKind::TyAlias if tcx.type_alias_is_lazy(def_id) => {
                terms_cx.add_inferreds_for_item(def_id)
            }
            _ => {}
        }
    }

    terms_cx
}

fn lang_items(tcx: TyCtxt<'_>) -> Vec<(LocalDefId, Vec<ty::Variance>)> {
    let lang_items = tcx.lang_items();
    let all = [
        (lang_items.phantom_data(), vec![ty::Covariant]),
        (lang_items.unsafe_cell_type(), vec![ty::Invariant]),
    ];

    all.into_iter() // iterating over (Option<DefId>, Variance)
        .filter_map(|(d, v)| {
            let def_id = d?.as_local()?; // LocalDefId
            Some((def_id, v))
        })
        .collect()
}

impl<'a, 'tcx> TermsContext<'a, 'tcx> {
    fn add_inferreds_for_item(&mut self, def_id: LocalDefId) {
        let tcx = self.tcx;
        let count = tcx.generics_of(def_id).count();

        if count == 0 {
            return;
        }

        // Record the start of this item's inferreds.
        let start = self.inferred_terms.len();
        let newly_added = self.inferred_starts.insert(def_id, InferredIndex(start)).is_none();
        assert!(newly_added);

        // N.B., in the code below for writing the results back into the
        // `CrateVariancesMap`, we rely on the fact that all inferreds
        // for a particular item are assigned continuous indices.

        let arena = self.arena;
        self.inferred_terms.extend(
            (start..(start + count)).map(|i| &*arena.alloc(InferredTerm(InferredIndex(i)))),
        );
    }
}
