use rustc_data_structures::sync::Lock;
use rustc_hir as hir;
use rustc_hir::def_id::LocalDefId;
use rustc_hir::intravisit;
use rustc_hir::{HirId, ItemLocalId};
use rustc_index::bit_set::GrowableBitSet;
use rustc_middle::hir::nested_filter;
use rustc_middle::ty::TyCtxt;

pub fn check_crate(tcx: TyCtxt<'_>) {
    tcx.dep_graph.assert_ignored();

    if tcx.sess.opts.unstable_opts.hir_stats {
        crate::hir_stats::print_hir_stats(tcx);
    }

    #[cfg(debug_assertions)]
    {
        let errors = Lock::new(Vec::new());

        tcx.hir().par_for_each_module(|module_id| {
            let mut v = HirIdValidator {
                tcx,
                owner: None,
                hir_ids_seen: Default::default(),
                errors: &errors,
            };

            tcx.hir().visit_item_likes_in_module(module_id, &mut v);
        });

        let errors = errors.into_inner();

        if !errors.is_empty() {
            let message = errors.iter().fold(String::new(), |s1, s2| s1 + "\n" + s2);
            tcx.sess.delay_span_bug(rustc_span::DUMMY_SP, &message);
        }
    }
}

struct HirIdValidator<'a, 'hir> {
    tcx: TyCtxt<'hir>,
    owner: Option<hir::OwnerId>,
    hir_ids_seen: GrowableBitSet<ItemLocalId>,
    errors: &'a Lock<Vec<String>>,
}

impl<'a, 'hir> HirIdValidator<'a, 'hir> {
    fn new_visitor(&self, tcx: TyCtxt<'hir>) -> HirIdValidator<'a, 'hir> {
        HirIdValidator { tcx, owner: None, hir_ids_seen: Default::default(), errors: self.errors }
    }

    #[cold]
    #[inline(never)]
    fn error(&self, f: impl FnOnce() -> String) {
        self.errors.lock().push(f());
    }

    fn check<F: FnOnce(&mut HirIdValidator<'a, 'hir>)>(&mut self, owner: hir::OwnerId, walk: F) {
        assert!(self.owner.is_none());
        self.owner = Some(owner);
        walk(self);

        if owner == hir::CRATE_OWNER_ID {
            return;
        }

        // There's always at least one entry for the owning item itself
        let max = self
            .hir_ids_seen
            .iter()
            .map(|local_id| local_id.as_usize())
            .max()
            .expect("owning item has no entry");

        if max != self.hir_ids_seen.len() - 1 {
            let hir = self.tcx.hir();
            let pretty_owner = hir.def_path(owner.def_id).to_string_no_crate_verbose();

            let missing_items: Vec<_> = (0..=max as u32)
                .map(|i| ItemLocalId::from_u32(i))
                .filter(|&local_id| !self.hir_ids_seen.contains(local_id))
                .map(|local_id| hir.node_to_string(HirId { owner, local_id }))
                .collect();

            let seen_items: Vec<_> = self
                .hir_ids_seen
                .iter()
                .map(|local_id| hir.node_to_string(HirId { owner, local_id }))
                .collect();

            self.error(|| {
                format!(
                    "ItemLocalIds not assigned densely in {}. \
                Max ItemLocalId = {}, missing IDs = {:#?}; seen IDs = {:#?}",
                    pretty_owner, max, missing_items, seen_items
                )
            });
        }
    }

    fn check_nested_id(&mut self, id: LocalDefId) {
        let Some(owner) = self.owner else { return };
        let def_parent = self.tcx.local_parent(id);
        let def_parent_hir_id = self.tcx.local_def_id_to_hir_id(def_parent);
        if def_parent_hir_id.owner != owner {
            self.error(|| {
                format!(
                    "inconsistent Def parent at `{:?}` for `{:?}`:\nexpected={:?}\nfound={:?}",
                    self.tcx.def_span(id),
                    id,
                    owner,
                    def_parent_hir_id
                )
            });
        }
    }
}

impl<'a, 'hir> intravisit::Visitor<'hir> for HirIdValidator<'a, 'hir> {
    type NestedFilter = nested_filter::OnlyBodies;

    fn nested_visit_map(&mut self) -> Self::Map {
        self.tcx.hir()
    }

    fn visit_nested_item(&mut self, id: hir::ItemId) {
        self.check_nested_id(id.owner_id.def_id);
    }

    fn visit_nested_trait_item(&mut self, id: hir::TraitItemId) {
        self.check_nested_id(id.owner_id.def_id);
    }

    fn visit_nested_impl_item(&mut self, id: hir::ImplItemId) {
        self.check_nested_id(id.owner_id.def_id);
    }

    fn visit_nested_foreign_item(&mut self, id: hir::ForeignItemId) {
        self.check_nested_id(id.owner_id.def_id);
    }

    fn visit_item(&mut self, i: &'hir hir::Item<'hir>) {
        let mut inner_visitor = self.new_visitor(self.tcx);
        inner_visitor.check(i.owner_id, |this| intravisit::walk_item(this, i));
    }

    fn visit_id(&mut self, hir_id: HirId) {
        let owner = self.owner.expect("no owner");

        if owner != hir_id.owner {
            self.error(|| {
                format!(
                    "HirIdValidator: The recorded owner of {} is {} instead of {}",
                    self.tcx.hir().node_to_string(hir_id),
                    self.tcx.hir().def_path(hir_id.owner.def_id).to_string_no_crate_verbose(),
                    self.tcx.hir().def_path(owner.def_id).to_string_no_crate_verbose()
                )
            });
        }

        self.hir_ids_seen.insert(hir_id.local_id);
    }

    fn visit_foreign_item(&mut self, i: &'hir hir::ForeignItem<'hir>) {
        let mut inner_visitor = self.new_visitor(self.tcx);
        inner_visitor.check(i.owner_id, |this| intravisit::walk_foreign_item(this, i));
    }

    fn visit_trait_item(&mut self, i: &'hir hir::TraitItem<'hir>) {
        let mut inner_visitor = self.new_visitor(self.tcx);
        inner_visitor.check(i.owner_id, |this| intravisit::walk_trait_item(this, i));
    }

    fn visit_impl_item(&mut self, i: &'hir hir::ImplItem<'hir>) {
        let mut inner_visitor = self.new_visitor(self.tcx);
        inner_visitor.check(i.owner_id, |this| intravisit::walk_impl_item(this, i));
    }
}
