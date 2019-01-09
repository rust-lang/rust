use hir::def_id::{DefId, DefIndex, CRATE_DEF_INDEX};
use hir::{self, intravisit, HirId, ItemLocalId};
use syntax::ast::NodeId;
use hir::itemlikevisit::ItemLikeVisitor;
use rustc_data_structures::fx::FxHashMap;

pub fn check_crate<'hir>(hir_map: &hir::map::Map<'hir>) {
    let mut outer_visitor = OuterVisitor {
        hir_map,
        errors: vec![],
    };

    hir_map.dep_graph.assert_ignored();

    hir_map.krate().visit_all_item_likes(&mut outer_visitor);
    if !outer_visitor.errors.is_empty() {
        let message = outer_visitor
            .errors
            .iter()
            .fold(String::new(), |s1, s2| s1 + "\n" + s2);
        bug!("{}", message);
    }
}

struct HirIdValidator<'a, 'hir: 'a> {
    hir_map: &'a hir::map::Map<'hir>,
    owner_def_index: Option<DefIndex>,
    hir_ids_seen: FxHashMap<ItemLocalId, NodeId>,
    errors: Vec<String>,
}

struct OuterVisitor<'a, 'hir: 'a> {
    hir_map: &'a hir::map::Map<'hir>,
    errors: Vec<String>,
}

impl<'a, 'hir: 'a> OuterVisitor<'a, 'hir> {
    fn new_inner_visitor(&self,
                         hir_map: &'a hir::map::Map<'hir>)
                         -> HirIdValidator<'a, 'hir> {
        HirIdValidator {
            hir_map,
            owner_def_index: None,
            hir_ids_seen: Default::default(),
            errors: Vec::new(),
        }
    }
}

impl<'a, 'hir: 'a> ItemLikeVisitor<'hir> for OuterVisitor<'a, 'hir> {
    fn visit_item(&mut self, i: &'hir hir::Item) {
        let mut inner_visitor = self.new_inner_visitor(self.hir_map);
        inner_visitor.check(i.id, |this| intravisit::walk_item(this, i));
        self.errors.extend(inner_visitor.errors.drain(..));
    }

    fn visit_trait_item(&mut self, i: &'hir hir::TraitItem) {
        let mut inner_visitor = self.new_inner_visitor(self.hir_map);
        inner_visitor.check(i.id, |this| intravisit::walk_trait_item(this, i));
        self.errors.extend(inner_visitor.errors.drain(..));
    }

    fn visit_impl_item(&mut self, i: &'hir hir::ImplItem) {
        let mut inner_visitor = self.new_inner_visitor(self.hir_map);
        inner_visitor.check(i.id, |this| intravisit::walk_impl_item(this, i));
        self.errors.extend(inner_visitor.errors.drain(..));
    }
}

impl<'a, 'hir: 'a> HirIdValidator<'a, 'hir> {

    fn check<F: FnOnce(&mut HirIdValidator<'a, 'hir>)>(&mut self,
                                                       node_id: NodeId,
                                                       walk: F) {
        assert!(self.owner_def_index.is_none());
        let owner_def_index = self.hir_map.local_def_id(node_id).index;
        self.owner_def_index = Some(owner_def_index);
        walk(self);

        if owner_def_index == CRATE_DEF_INDEX {
            return;
        }

        // There's always at least one entry for the owning item itself
        let max = self.hir_ids_seen
                      .keys()
                      .map(|local_id| local_id.as_usize())
                      .max()
                      .expect("owning item has no entry");

        if max != self.hir_ids_seen.len() - 1 {
            // Collect the missing ItemLocalIds
            let missing: Vec<_> = (0 ..= max as u32)
              .filter(|&i| !self.hir_ids_seen.contains_key(&ItemLocalId::from_u32(i)))
              .collect();

            // Try to map those to something more useful
            let mut missing_items = Vec::with_capacity(missing.len());

            for local_id in missing {
                let hir_id = HirId {
                    owner: owner_def_index,
                    local_id: ItemLocalId::from_u32(local_id),
                };

                trace!("missing hir id {:#?}", hir_id);

                // We are already in ICE mode here, so doing a linear search
                // should be fine.
                let (node_id, _) = self.hir_map
                                       .definitions()
                                       .node_to_hir_id
                                       .iter()
                                       .enumerate()
                                       .find(|&(_, &entry)| hir_id == entry)
                                       .expect("no node_to_hir_id entry");
                let node_id = NodeId::from_usize(node_id);
                missing_items.push(format!("[local_id: {}, node:{}]",
                                           local_id,
                                           self.hir_map.node_to_string(node_id)));
            }
            self.errors.push(format!(
                "ItemLocalIds not assigned densely in {}. \
                Max ItemLocalId = {}, missing IDs = {:?}; seens IDs = {:?}",
                self.hir_map.def_path(DefId::local(owner_def_index)).to_string_no_crate(),
                max,
                missing_items,
                self.hir_ids_seen
                    .values()
                    .map(|n| format!("({:?} {})", n, self.hir_map.node_to_string(*n)))
                    .collect::<Vec<_>>()));
        }
    }
}

impl<'a, 'hir: 'a> intravisit::Visitor<'hir> for HirIdValidator<'a, 'hir> {

    fn nested_visit_map<'this>(&'this mut self)
                               -> intravisit::NestedVisitorMap<'this, 'hir> {
        intravisit::NestedVisitorMap::OnlyBodies(self.hir_map)
    }

    fn visit_id(&mut self, node_id: NodeId) {
        let owner = self.owner_def_index.expect("no owner_def_index");
        let stable_id = self.hir_map.definitions().node_to_hir_id[node_id];

        if stable_id == hir::DUMMY_HIR_ID {
            self.errors.push(format!("HirIdValidator: No HirId assigned for NodeId {}: {:?}",
                                     node_id,
                                     self.hir_map.node_to_string(node_id)));
            return;
        }

        if owner != stable_id.owner {
            self.errors.push(format!(
                "HirIdValidator: The recorded owner of {} is {} instead of {}",
                self.hir_map.node_to_string(node_id),
                self.hir_map.def_path(DefId::local(stable_id.owner)).to_string_no_crate(),
                self.hir_map.def_path(DefId::local(owner)).to_string_no_crate()));
        }

        if let Some(prev) = self.hir_ids_seen.insert(stable_id.local_id, node_id) {
            if prev != node_id {
                self.errors.push(format!(
                    "HirIdValidator: Same HirId {}/{} assigned for nodes {} and {}",
                    self.hir_map.def_path(DefId::local(stable_id.owner)).to_string_no_crate(),
                    stable_id.local_id.as_usize(),
                    self.hir_map.node_to_string(prev),
                    self.hir_map.node_to_string(node_id)));
            }
        }
    }

    fn visit_impl_item_ref(&mut self, _: &'hir hir::ImplItemRef) {
        // Explicitly do nothing here. ImplItemRefs contain hir::Visibility
        // values that actually belong to an ImplItem instead of the ItemKind::Impl
        // we are currently in. So for those it's correct that they have a
        // different owner.
    }
}
