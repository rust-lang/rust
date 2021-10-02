use rustc_data_structures::fingerprint::Fingerprint;
use rustc_data_structures::stable_hasher::{HashStable, StableHasher};
use rustc_data_structures::svh::Svh;
use rustc_hir::def_id::{CrateNum, LocalDefId, StableCrateId, LOCAL_CRATE};
use rustc_hir::intravisit::{self, Visitor};
use rustc_hir::{ForeignItem, ImplItem, Item, ItemKind, Mod, TraitItem};
use rustc_hir::{ForeignItemId, HirId, ImplItemId, ItemId, ModuleItems, TraitItemId};
use rustc_middle::hir::nested_filter;
use rustc_middle::ty::query::Providers;
use rustc_middle::ty::TyCtxt;
use rustc_span::Span;

pub fn provide(providers: &mut Providers) {
    *providers = Providers { crate_hash, hir_module_items, hir_crate_items, ..*providers };
}

fn crate_hash(tcx: TyCtxt<'_>, crate_num: CrateNum) -> Svh {
    debug_assert_eq!(crate_num, LOCAL_CRATE);
    let krate = tcx.hir_crate(());
    let hir_body_hash = krate.hir_hash;

    let upstream_crates = upstream_crates(tcx);

    // We hash the final, remapped names of all local source files so we
    // don't have to include the path prefix remapping commandline args.
    // If we included the full mapping in the SVH, we could only have
    // reproducible builds by compiling from the same directory. So we just
    // hash the result of the mapping instead of the mapping itself.
    let mut source_file_names: Vec<_> = tcx
        .sess
        .source_map()
        .files()
        .iter()
        .filter(|source_file| source_file.cnum == LOCAL_CRATE)
        .map(|source_file| source_file.name_hash)
        .collect();

    source_file_names.sort_unstable();

    let crate_hash: Fingerprint = tcx.with_stable_hashing_context(|mut hcx| {
        let mut stable_hasher = StableHasher::new();
        hir_body_hash.hash_stable(&mut hcx, &mut stable_hasher);
        upstream_crates.hash_stable(&mut hcx, &mut stable_hasher);
        source_file_names.hash_stable(&mut hcx, &mut stable_hasher);
        if tcx.sess.opts.debugging_opts.incremental_relative_spans {
            let definitions = tcx.definitions_untracked();
            let mut owner_spans: Vec<_> = krate
                .owners
                .iter_enumerated()
                .filter_map(|(def_id, info)| {
                    let _ = info.as_owner()?;
                    let def_path_hash = definitions.def_path_hash(def_id);
                    let span = definitions.def_span(def_id);
                    debug_assert_eq!(span.parent(), None);
                    Some((def_path_hash, span))
                })
                .collect();
            owner_spans.sort_unstable_by_key(|bn| bn.0);
            owner_spans.hash_stable(&mut hcx, &mut stable_hasher);
        }
        tcx.sess.opts.dep_tracking_hash(true).hash_stable(&mut hcx, &mut stable_hasher);
        tcx.sess.local_stable_crate_id().hash_stable(&mut hcx, &mut stable_hasher);
        // Hash visibility information since it does not appear in HIR.
        let resolutions = tcx.resolutions(());
        resolutions.visibilities.hash_stable(&mut hcx, &mut stable_hasher);
        resolutions.has_pub_restricted.hash_stable(&mut hcx, &mut stable_hasher);
        stable_hasher.finish()
    });

    Svh::new(crate_hash.to_smaller_hash())
}

fn upstream_crates(tcx: TyCtxt<'_>) -> Vec<(StableCrateId, Svh)> {
    let mut upstream_crates: Vec<_> = tcx
        .crates(())
        .iter()
        .map(|&cnum| {
            let stable_crate_id = tcx.stable_crate_id(cnum);
            let hash = tcx.crate_hash(cnum);
            (stable_crate_id, hash)
        })
        .collect();
    upstream_crates.sort_unstable_by_key(|&(stable_crate_id, _)| stable_crate_id);
    upstream_crates
}

fn hir_module_items(tcx: TyCtxt<'_>, module_id: LocalDefId) -> ModuleItems {
    let mut collector = ModuleCollector {
        tcx,
        submodules: Vec::default(),
        items: Vec::default(),
        trait_items: Vec::default(),
        impl_items: Vec::default(),
        foreign_items: Vec::default(),
    };

    let (hir_mod, span, hir_id) = tcx.hir().get_module(module_id);
    collector.visit_mod(hir_mod, span, hir_id);

    let ModuleCollector { submodules, items, trait_items, impl_items, foreign_items, .. } =
        collector;
    return ModuleItems {
        submodules: submodules.into_boxed_slice(),
        items: items.into_boxed_slice(),
        trait_items: trait_items.into_boxed_slice(),
        impl_items: impl_items.into_boxed_slice(),
        foreign_items: foreign_items.into_boxed_slice(),
    };

    struct ModuleCollector<'tcx> {
        tcx: TyCtxt<'tcx>,
        submodules: Vec<LocalDefId>,
        items: Vec<ItemId>,
        trait_items: Vec<TraitItemId>,
        impl_items: Vec<ImplItemId>,
        foreign_items: Vec<ForeignItemId>,
    }

    impl<'hir> Visitor<'hir> for ModuleCollector<'hir> {
        type NestedFilter = nested_filter::All;

        fn nested_visit_map(&mut self) -> Self::Map {
            self.tcx.hir()
        }

        fn visit_item(&mut self, item: &'hir Item<'hir>) {
            self.items.push(item.item_id());
            if let ItemKind::Mod(..) = item.kind {
                // If this declares another module, do not recurse inside it.
                self.submodules.push(item.def_id);
            } else {
                intravisit::walk_item(self, item)
            }
        }

        fn visit_trait_item(&mut self, item: &'hir TraitItem<'hir>) {
            self.trait_items.push(item.trait_item_id());
            intravisit::walk_trait_item(self, item)
        }

        fn visit_impl_item(&mut self, item: &'hir ImplItem<'hir>) {
            self.impl_items.push(item.impl_item_id());
            intravisit::walk_impl_item(self, item)
        }

        fn visit_foreign_item(&mut self, item: &'hir ForeignItem<'hir>) {
            self.foreign_items.push(item.foreign_item_id());
            intravisit::walk_foreign_item(self, item)
        }
    }
}

fn hir_crate_items(tcx: TyCtxt<'_>, _: ()) -> ModuleItems {
    let mut collector = CrateCollector {
        tcx,
        submodules: Vec::default(),
        items: Vec::default(),
        trait_items: Vec::default(),
        impl_items: Vec::default(),
        foreign_items: Vec::default(),
    };

    tcx.hir().walk_toplevel_module(&mut collector);

    let CrateCollector { submodules, items, trait_items, impl_items, foreign_items, .. } =
        collector;

    return ModuleItems {
        submodules: submodules.into_boxed_slice(),
        items: items.into_boxed_slice(),
        trait_items: trait_items.into_boxed_slice(),
        impl_items: impl_items.into_boxed_slice(),
        foreign_items: foreign_items.into_boxed_slice(),
    };

    struct CrateCollector<'tcx> {
        tcx: TyCtxt<'tcx>,
        submodules: Vec<LocalDefId>,
        items: Vec<ItemId>,
        trait_items: Vec<TraitItemId>,
        impl_items: Vec<ImplItemId>,
        foreign_items: Vec<ForeignItemId>,
    }

    impl<'hir> Visitor<'hir> for CrateCollector<'hir> {
        type NestedFilter = nested_filter::All;

        fn nested_visit_map(&mut self) -> Self::Map {
            self.tcx.hir()
        }

        fn visit_item(&mut self, item: &'hir Item<'hir>) {
            self.items.push(item.item_id());
            intravisit::walk_item(self, item)
        }

        fn visit_mod(&mut self, m: &'hir Mod<'hir>, _s: Span, n: HirId) {
            self.submodules.push(n.owner);
            intravisit::walk_mod(self, m, n);
        }

        fn visit_foreign_item(&mut self, item: &'hir ForeignItem<'hir>) {
            self.foreign_items.push(item.foreign_item_id());
            intravisit::walk_foreign_item(self, item)
        }

        fn visit_trait_item(&mut self, item: &'hir TraitItem<'hir>) {
            self.trait_items.push(item.trait_item_id());
            intravisit::walk_trait_item(self, item)
        }

        fn visit_impl_item(&mut self, item: &'hir ImplItem<'hir>) {
            self.impl_items.push(item.impl_item_id());
            intravisit::walk_impl_item(self, item)
        }
    }
}
