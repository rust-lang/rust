use rustc_data_structures::fx::FxHashSet;
use rustc_data_structures::sync::Lock;
use rustc_hir as hir;
use rustc_hir::def_id::{LocalDefId, CRATE_DEF_INDEX};
use rustc_hir::intravisit;
use rustc_hir::itemlikevisit::ItemLikeVisitor;
use rustc_hir::{HirId, ItemLocalId};
use rustc_middle::hir::map::Map;
use rustc_middle::ty::TyCtxt;

pub fn check_crate(tcx: TyCtxt<'_>) {
    tcx.dep_graph.assert_ignored();

    if tcx.sess.opts.debugging_opts.hir_stats {
        crate::hir_stats::print_hir_stats(tcx);
    }

    let errors = Lock::new(Vec::new());
    let hir_map = tcx.hir();

    hir_map.par_for_each_module(|module_id| {
        hir_map
            .visit_item_likes_in_module(module_id, &mut OuterVisitor { hir_map, errors: &errors })
    });

    let errors = errors.into_inner();

    if !errors.is_empty() {
        let message = errors.iter().fold(String::new(), |s1, s2| s1 + "\n" + s2);
        tcx.sess.delay_span_bug(rustc_span::DUMMY_SP, &message);
    }
}

struct HirIdValidator<'a, 'hir> {
    hir_map: Map<'hir>,
    owner: Option<LocalDefId>,
    hir_ids_seen: FxHashSet<ItemLocalId>,
    errors: &'a Lock<Vec<String>>,
}

struct OuterVisitor<'a, 'hir> {
    hir_map: Map<'hir>,
    errors: &'a Lock<Vec<String>>,
}

impl<'a, 'hir> OuterVisitor<'a, 'hir> {
    fn new_inner_visitor(&self, hir_map: Map<'hir>) -> HirIdValidator<'a, 'hir> {
        HirIdValidator {
            hir_map,
            owner: None,
            hir_ids_seen: Default::default(),
            errors: self.errors,
        }
    }
}

impl<'a, 'hir> ItemLikeVisitor<'hir> for OuterVisitor<'a, 'hir> {
    fn visit_item(&mut self, i: &'hir hir::Item<'hir>) {
        let mut inner_visitor = self.new_inner_visitor(self.hir_map);
        inner_visitor.check(i.hir_id(), |this| intravisit::walk_item(this, i));
    }

    fn visit_trait_item(&mut self, i: &'hir hir::TraitItem<'hir>) {
        let mut inner_visitor = self.new_inner_visitor(self.hir_map);
        inner_visitor.check(i.hir_id(), |this| intravisit::walk_trait_item(this, i));
    }

    fn visit_impl_item(&mut self, i: &'hir hir::ImplItem<'hir>) {
        let mut inner_visitor = self.new_inner_visitor(self.hir_map);
        inner_visitor.check(i.hir_id(), |this| intravisit::walk_impl_item(this, i));
    }

    fn visit_foreign_item(&mut self, i: &'hir hir::ForeignItem<'hir>) {
        let mut inner_visitor = self.new_inner_visitor(self.hir_map);
        inner_visitor.check(i.hir_id(), |this| intravisit::walk_foreign_item(this, i));
    }
}

impl<'a, 'hir> HirIdValidator<'a, 'hir> {
    #[cold]
    #[inline(never)]
    fn error(&self, f: impl FnOnce() -> String) {
        self.errors.lock().push(f());
    }

    fn check<F: FnOnce(&mut HirIdValidator<'a, 'hir>)>(&mut self, hir_id: HirId, walk: F) {
        assert!(self.owner.is_none());
        let owner = self.hir_map.local_def_id(hir_id);
        self.owner = Some(owner);
        walk(self);

        if owner.local_def_index == CRATE_DEF_INDEX {
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
            // Collect the missing ItemLocalIds
            let missing: Vec<_> = (0..=max as u32)
                .filter(|&i| !self.hir_ids_seen.contains(&ItemLocalId::from_u32(i)))
                .collect();

            // Try to map those to something more useful
            let mut missing_items = Vec::with_capacity(missing.len());

            for local_id in missing {
                let hir_id = HirId { owner, local_id: ItemLocalId::from_u32(local_id) };

                trace!("missing hir id {:#?}", hir_id);

                missing_items.push(format!(
                    "[local_id: {}, owner: {}]",
                    local_id,
                    self.hir_map.def_path(owner).to_string_no_crate_verbose()
                ));
            }
            self.error(|| {
                format!(
                    "ItemLocalIds not assigned densely in {}. \
                Max ItemLocalId = {}, missing IDs = {:?}; seens IDs = {:?}",
                    self.hir_map.def_path(owner).to_string_no_crate_verbose(),
                    max,
                    missing_items,
                    self.hir_ids_seen
                        .iter()
                        .map(|&local_id| HirId { owner, local_id })
                        .map(|h| format!("({:?} {})", h, self.hir_map.node_to_string(h)))
                        .collect::<Vec<_>>()
                )
            });
        }
    }
}

impl<'a, 'hir> intravisit::Visitor<'hir> for HirIdValidator<'a, 'hir> {
    type Map = Map<'hir>;

    fn nested_visit_map(&mut self) -> intravisit::NestedVisitorMap<Self::Map> {
        intravisit::NestedVisitorMap::OnlyBodies(self.hir_map)
    }

    fn visit_id(&mut self, hir_id: HirId) {
        let owner = self.owner.expect("no owner");

        if owner != hir_id.owner {
            self.error(|| {
                format!(
                    "HirIdValidator: The recorded owner of {} is {} instead of {}",
                    self.hir_map.node_to_string(hir_id),
                    self.hir_map.def_path(hir_id.owner).to_string_no_crate_verbose(),
                    self.hir_map.def_path(owner).to_string_no_crate_verbose()
                )
            });
        }

        self.hir_ids_seen.insert(hir_id.local_id);
    }

    fn visit_impl_item_ref(&mut self, _: &'hir hir::ImplItemRef) {
        // Explicitly do nothing here. ImplItemRefs contain hir::Visibility
        // values that actually belong to an ImplItem instead of the ItemKind::Impl
        // we are currently in. So for those it's correct that they have a
        // different owner.
    }

    fn visit_foreign_item_ref(&mut self, _: &'hir hir::ForeignItemRef) {
        // Explicitly do nothing here. ForeignItemRefs contain hir::Visibility
        // values that actually belong to an ForeignItem instead of the ItemKind::ForeignMod
        // we are currently in. So for those it's correct that they have a
        // different owner.
    }
}
