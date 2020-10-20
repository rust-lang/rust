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

use rustc_arena::DroplessArena;
use rustc_hir as hir;
use rustc_hir::itemlikevisit::ItemLikeVisitor;
use rustc_hir::HirIdMap;
use rustc_middle::ty::{self, TyCtxt};
use std::fmt;

use self::VarianceTerm::*;

pub type VarianceTermPtr<'a> = &'a VarianceTerm<'a>;

#[derive(Copy, Clone, Debug)]
pub struct InferredIndex(pub usize);

#[derive(Copy, Clone)]
pub enum VarianceTerm<'a> {
    ConstantTerm(ty::Variance),
    TransformTerm(VarianceTermPtr<'a>, VarianceTermPtr<'a>),
    InferredTerm(InferredIndex),
}

impl<'a> fmt::Debug for VarianceTerm<'a> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match *self {
            ConstantTerm(c1) => write!(f, "{:?}", c1),
            TransformTerm(v1, v2) => write!(f, "({:?} \u{00D7} {:?})", v1, v2),
            InferredTerm(id) => write!(f, "[{}]", {
                let InferredIndex(i) = id;
                i
            }),
        }
    }
}

// The first pass over the crate simply builds up the set of inferreds.

pub struct TermsContext<'a, 'tcx> {
    pub tcx: TyCtxt<'tcx>,
    pub arena: &'a DroplessArena,

    // For marker types, UnsafeCell, and other lang items where
    // variance is hardcoded, records the item-id and the hardcoded
    // variance.
    pub lang_items: Vec<(hir::HirId, Vec<ty::Variance>)>,

    // Maps from the node id of an item to the first inferred index
    // used for its type & region parameters.
    pub inferred_starts: HirIdMap<InferredIndex>,

    // Maps from an InferredIndex to the term for that variable.
    pub inferred_terms: Vec<VarianceTermPtr<'a>>,
}

pub fn determine_parameters_to_be_inferred<'a, 'tcx>(
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
    tcx.hir().krate().visit_all_item_likes(&mut terms_cx);

    terms_cx
}

fn lang_items(tcx: TyCtxt<'_>) -> Vec<(hir::HirId, Vec<ty::Variance>)> {
    let lang_items = tcx.lang_items();
    let all = vec![
        (lang_items.phantom_data(), vec![ty::Covariant]),
        (lang_items.unsafe_cell_type(), vec![ty::Invariant]),
    ];

    all.into_iter() // iterating over (Option<DefId>, Variance)
        .filter(|&(ref d, _)| d.is_some())
        .map(|(d, v)| (d.unwrap(), v)) // (DefId, Variance)
        .filter_map(|(d, v)| {
            d.as_local().map(|d| tcx.hir().local_def_id_to_hir_id(d)).map(|n| (n, v))
        }) // (HirId, Variance)
        .collect()
}

impl<'a, 'tcx> TermsContext<'a, 'tcx> {
    fn add_inferreds_for_item(&mut self, id: hir::HirId) {
        let tcx = self.tcx;
        let def_id = tcx.hir().local_def_id(id);
        let count = tcx.generics_of(def_id).count();

        if count == 0 {
            return;
        }

        // Record the start of this item's inferreds.
        let start = self.inferred_terms.len();
        let newly_added = self.inferred_starts.insert(id, InferredIndex(start)).is_none();
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

impl<'a, 'tcx, 'v> ItemLikeVisitor<'v> for TermsContext<'a, 'tcx> {
    fn visit_item(&mut self, item: &hir::Item<'_>) {
        debug!("add_inferreds for item {}", self.tcx.hir().node_to_string(item.hir_id));

        match item.kind {
            hir::ItemKind::Struct(ref struct_def, _) | hir::ItemKind::Union(ref struct_def, _) => {
                self.add_inferreds_for_item(item.hir_id);

                if let hir::VariantData::Tuple(..) = *struct_def {
                    self.add_inferreds_for_item(struct_def.ctor_hir_id().unwrap());
                }
            }

            hir::ItemKind::Enum(ref enum_def, _) => {
                self.add_inferreds_for_item(item.hir_id);

                for variant in enum_def.variants {
                    if let hir::VariantData::Tuple(..) = variant.data {
                        self.add_inferreds_for_item(variant.data.ctor_hir_id().unwrap());
                    }
                }
            }

            hir::ItemKind::Fn(..) => {
                self.add_inferreds_for_item(item.hir_id);
            }

            hir::ItemKind::ForeignMod(ref foreign_mod) => {
                for foreign_item in foreign_mod.items {
                    if let hir::ForeignItemKind::Fn(..) = foreign_item.kind {
                        self.add_inferreds_for_item(foreign_item.hir_id);
                    }
                }
            }

            _ => {}
        }
    }

    fn visit_trait_item(&mut self, trait_item: &hir::TraitItem<'_>) {
        if let hir::TraitItemKind::Fn(..) = trait_item.kind {
            self.add_inferreds_for_item(trait_item.hir_id);
        }
    }

    fn visit_impl_item(&mut self, impl_item: &hir::ImplItem<'_>) {
        if let hir::ImplItemKind::Fn(..) = impl_item.kind {
            self.add_inferreds_for_item(impl_item.hir_id);
        }
    }
}
