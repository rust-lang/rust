use crate::hir::def_id::{DefId, DefIndex, CRATE_DEF_INDEX};
use crate::hir::{self, intravisit, HirId, ItemLocalId};
use crate::hir::itemlikevisit::ItemLikeVisitor;
use rustc_data_structures::fx::FxHashSet;
use rustc_data_structures::sync::{Lock, ParallelIterator, par_iter};

pub fn check_crate(hir_map: &hir::map::Map<'_>) {
    hir_map.dep_graph.assert_ignored();

    let errors = Lock::new(Vec::new());

    par_iter(&hir_map.krate().modules).for_each(|(module_id, _)| {
        hir_map.visit_item_likes_in_module(hir_map.local_def_id(*module_id), &mut OuterVisitor {
            hir_map,
            errors: &errors,
        });
    });

    let errors = errors.into_inner();

    if !errors.is_empty() {
        let message = errors
            .iter()
            .fold(String::new(), |s1, s2| s1 + "\n" + s2);
        bug!("{}", message);
    }
}

struct HirIdValidator<'a, 'hir> {
    hir_map: &'a hir::map::Map<'hir>,
    owner_def_index: Option<DefIndex>,
    hir_ids_seen: FxHashSet<ItemLocalId>,
    errors: &'a Lock<Vec<String>>,
}

struct OuterVisitor<'a, 'hir> {
    hir_map: &'a hir::map::Map<'hir>,
    errors: &'a Lock<Vec<String>>,
}

impl<'a, 'hir> OuterVisitor<'a, 'hir> {
    fn new_inner_visitor(&self,
                         hir_map: &'a hir::map::Map<'hir>)
                         -> HirIdValidator<'a, 'hir> {
        HirIdValidator {
            hir_map,
            owner_def_index: None,
            hir_ids_seen: Default::default(),
            errors: self.errors,
        }
    }
}

impl<'a, 'hir> ItemLikeVisitor<'hir> for OuterVisitor<'a, 'hir> {
    fn visit_item(&mut self, i: &'hir hir::Item) {
        let mut inner_visitor = self.new_inner_visitor(self.hir_map);
        inner_visitor.check(i.hir_id, |this| intravisit::walk_item(this, i));
    }

    fn visit_trait_item(&mut self, i: &'hir hir::TraitItem) {
        let mut inner_visitor = self.new_inner_visitor(self.hir_map);
        inner_visitor.check(i.hir_id, |this| intravisit::walk_trait_item(this, i));
    }

    fn visit_impl_item(&mut self, i: &'hir hir::ImplItem) {
        let mut inner_visitor = self.new_inner_visitor(self.hir_map);
        inner_visitor.check(i.hir_id, |this| intravisit::walk_impl_item(this, i));
    }
}

impl<'a, 'hir> HirIdValidator<'a, 'hir> {
    #[cold]
    #[inline(never)]
    fn error(&self, f: impl FnOnce() -> String) {
        self.errors.lock().push(f());
    }

    fn check<F: FnOnce(&mut HirIdValidator<'a, 'hir>)>(&mut self,
                                                       hir_id: HirId,
                                                       walk: F) {
        assert!(self.owner_def_index.is_none());
        let owner_def_index = self.hir_map.local_def_id_from_hir_id(hir_id).index;
        self.owner_def_index = Some(owner_def_index);
        walk(self);

        if owner_def_index == CRATE_DEF_INDEX {
            return;
        }

        // There's always at least one entry for the owning item itself
        let max = self.hir_ids_seen
                      .iter()
                      .map(|local_id| local_id.as_usize())
                      .max()
                      .expect("owning item has no entry");

        if max != self.hir_ids_seen.len() - 1 {
            // Collect the missing ItemLocalIds
            let missing: Vec<_> = (0 ..= max as u32)
              .filter(|&i| !self.hir_ids_seen.contains(&ItemLocalId::from_u32(i)))
              .collect();

            // Try to map those to something more useful
            let mut missing_items = Vec::with_capacity(missing.len());

            for local_id in missing {
                let hir_id = HirId {
                    owner: owner_def_index,
                    local_id: ItemLocalId::from_u32(local_id),
                };

                trace!("missing hir id {:#?}", hir_id);

                missing_items.push(format!("[local_id: {}, node:{}]",
                                           local_id,
                                           self.hir_map.node_to_string(hir_id)));
            }
            self.error(|| format!(
                "ItemLocalIds not assigned densely in {}. \
                Max ItemLocalId = {}, missing IDs = {:?}; seens IDs = {:?}",
                self.hir_map.def_path(DefId::local(owner_def_index)).to_string_no_crate(),
                max,
                missing_items,
                self.hir_ids_seen
                    .iter()
                    .map(|&local_id| HirId {
                        owner: owner_def_index,
                        local_id,
                    })
                    .map(|h| format!("({:?} {})", h, self.hir_map.node_to_string(h)))
                    .collect::<Vec<_>>()));
        }
    }
}

impl<'a, 'hir> intravisit::Visitor<'hir> for HirIdValidator<'a, 'hir> {

    fn nested_visit_map<'this>(&'this mut self)
                               -> intravisit::NestedVisitorMap<'this, 'hir> {
        intravisit::NestedVisitorMap::OnlyBodies(self.hir_map)
    }

    fn visit_id(&mut self, hir_id: HirId) {
        let owner = self.owner_def_index.expect("no owner_def_index");

        if hir_id == hir::DUMMY_HIR_ID {
            self.error(|| format!("HirIdValidator: HirId {:?} is invalid",
                                  self.hir_map.node_to_string(hir_id)));
            return;
        }

        if owner != hir_id.owner {
            self.error(|| format!(
                "HirIdValidator: The recorded owner of {} is {} instead of {}",
                self.hir_map.node_to_string(hir_id),
                self.hir_map.def_path(DefId::local(hir_id.owner)).to_string_no_crate(),
                self.hir_map.def_path(DefId::local(owner)).to_string_no_crate()));
        }

        self.hir_ids_seen.insert(hir_id.local_id);
    }

    fn visit_impl_item_ref(&mut self, _: &'hir hir::ImplItemRef) {
        // Explicitly do nothing here. ImplItemRefs contain hir::Visibility
        // values that actually belong to an ImplItem instead of the ItemKind::Impl
        // we are currently in. So for those it's correct that they have a
        // different owner.
    }
}
