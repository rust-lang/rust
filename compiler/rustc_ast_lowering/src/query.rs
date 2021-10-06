use rustc_data_structures::fingerprint::Fingerprint;
use rustc_data_structures::stable_hasher::{HashStable, StableHasher};
use rustc_data_structures::svh::Svh;
use rustc_hir::def_id::{LocalDefId, LocalModDefId, StableCrateId, LOCAL_CRATE};
use rustc_hir::intravisit::{self, Visitor};
use rustc_hir::*;
use rustc_middle::hir::{nested_filter, ModuleItems};
use rustc_middle::middle::debugger_visualizer::DebuggerVisualizerFile;
use rustc_middle::query::{LocalCrate, Providers};
use rustc_middle::ty::TyCtxt;
use rustc_span::DUMMY_SP;

use crate::{index_ast, lower_to_hir};

pub fn provide(providers: &mut Providers) {
    *providers = Providers {
        lower_to_hir,
        index_ast,
        crate_hash,
        hir_module_items,
        hir_crate_items,
        ..*providers
    };
}

fn crate_hash(tcx: TyCtxt<'_>, _: LocalCrate) -> Svh {
    let mut hir_body_nodes: Vec<_> = tcx
        .hir_crate_items(())
        .owners()
        .map(|owner| {
            let def_id = owner.def_id;
            let def_path_hash = tcx.hir().def_path_hash(def_id);
            let info = tcx.lower_to_hir(def_id).unwrap();
            let span = if tcx.sess.opts.incremental.is_some() {
                tcx.source_span(def_id)
            } else {
                DUMMY_SP
            };
            debug_assert_eq!(span.parent(), None);
            (def_path_hash, info, span)
        })
        .collect();
    hir_body_nodes.sort_unstable_by_key(|bn| bn.0);

    let upstream_crates = upstream_crates(tcx);

    let resolutions = tcx.resolutions(());

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
        .map(|source_file| source_file.stable_id)
        .collect();

    source_file_names.sort_unstable();

    // We have to take care of debugger visualizers explicitly. The HIR (and
    // thus `hir_body_hash`) contains the #[debugger_visualizer] attributes but
    // these attributes only store the file path to the visualizer file, not
    // their content. Yet that content is exported into crate metadata, so any
    // changes to it need to be reflected in the crate hash.
    let debugger_visualizers: Vec<_> = tcx
        .debugger_visualizers(LOCAL_CRATE)
        .iter()
        // We ignore the path to the visualizer file since it's not going to be
        // encoded in crate metadata and we already hash the full contents of
        // the file.
        .map(DebuggerVisualizerFile::path_erased)
        .collect();

    let crate_hash: Fingerprint = tcx.with_stable_hashing_context(|mut hcx| {
        let mut stable_hasher = StableHasher::new();
        hir_body_nodes.hash_stable(&mut hcx, &mut stable_hasher);
        upstream_crates.hash_stable(&mut hcx, &mut stable_hasher);
        source_file_names.hash_stable(&mut hcx, &mut stable_hasher);
        debugger_visualizers.hash_stable(&mut hcx, &mut stable_hasher);
        tcx.sess.opts.dep_tracking_hash(true).hash_stable(&mut hcx, &mut stable_hasher);
        tcx.stable_crate_id(LOCAL_CRATE).hash_stable(&mut hcx, &mut stable_hasher);
        // Hash visibility information since it does not appear in HIR.
        // FIXME: Figure out how to remove `visibilities_for_hashing` by hashing visibilities on
        // the fly in the resolver, storing only their accumulated hash in `ResolverGlobalCtxt`,
        // and combining it with other hashes here.
        resolutions.visibilities_for_hashing.hash_stable(&mut hcx, &mut stable_hasher);
        stable_hasher.finish()
    });

    Svh::new(crate_hash)
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

fn hir_module_items(tcx: TyCtxt<'_>, module_id: LocalModDefId) -> ModuleItems {
    let mut collector = ItemCollector::new(tcx, false);

    let (hir_mod, span, hir_id) = tcx.hir().get_module(module_id);
    collector.visit_mod(hir_mod, span, hir_id);

    let ItemCollector {
        submodules,
        items,
        trait_items,
        impl_items,
        foreign_items,
        body_owners,
        ..
    } = collector;
    return ModuleItems {
        add_root: false,
        submodules: submodules.into_boxed_slice(),
        free_items: items.into_boxed_slice(),
        trait_items: trait_items.into_boxed_slice(),
        impl_items: impl_items.into_boxed_slice(),
        foreign_items: foreign_items.into_boxed_slice(),
        body_owners: body_owners.into_boxed_slice(),
    };
}

fn hir_crate_items(tcx: TyCtxt<'_>, _: ()) -> ModuleItems {
    let mut collector = ItemCollector::new(tcx, true);

    // A "crate collector" and "module collector" start at a
    // module item (the former starts at the crate root) but only
    // the former needs to collect it. ItemCollector does not do this for us.
    collector.submodules.push(CRATE_OWNER_ID);
    tcx.hir().walk_toplevel_module(&mut collector);

    let ItemCollector {
        submodules,
        items,
        trait_items,
        impl_items,
        foreign_items,
        body_owners,
        ..
    } = collector;

    return ModuleItems {
        add_root: true,
        submodules: submodules.into_boxed_slice(),
        free_items: items.into_boxed_slice(),
        trait_items: trait_items.into_boxed_slice(),
        impl_items: impl_items.into_boxed_slice(),
        foreign_items: foreign_items.into_boxed_slice(),
        body_owners: body_owners.into_boxed_slice(),
    };
}

struct ItemCollector<'tcx> {
    // When true, it collects all items in the create,
    // otherwise it collects items in some module.
    crate_collector: bool,
    tcx: TyCtxt<'tcx>,
    submodules: Vec<OwnerId>,
    items: Vec<ItemId>,
    trait_items: Vec<TraitItemId>,
    impl_items: Vec<ImplItemId>,
    foreign_items: Vec<ForeignItemId>,
    body_owners: Vec<LocalDefId>,
}

impl<'tcx> ItemCollector<'tcx> {
    fn new(tcx: TyCtxt<'tcx>, crate_collector: bool) -> ItemCollector<'tcx> {
        ItemCollector {
            crate_collector,
            tcx,
            submodules: Vec::default(),
            items: Vec::default(),
            trait_items: Vec::default(),
            impl_items: Vec::default(),
            foreign_items: Vec::default(),
            body_owners: Vec::default(),
        }
    }
}

impl<'hir> Visitor<'hir> for ItemCollector<'hir> {
    type NestedFilter = nested_filter::All;

    fn nested_visit_map(&mut self) -> Self::Map {
        self.tcx.hir()
    }

    fn visit_item(&mut self, item: &'hir Item<'hir>) {
        if Node::Item(item).associated_body().is_some() {
            self.body_owners.push(item.owner_id.def_id);
        }

        self.items.push(item.item_id());

        // Items that are modules are handled here instead of in visit_mod.
        if let ItemKind::Mod(module) = &item.kind {
            self.submodules.push(item.owner_id);
            // A module collector does not recurse inside nested modules.
            if self.crate_collector {
                intravisit::walk_mod(self, module, item.hir_id());
            }
        } else {
            intravisit::walk_item(self, item)
        }
    }

    fn visit_foreign_item(&mut self, item: &'hir ForeignItem<'hir>) {
        self.foreign_items.push(item.foreign_item_id());
        intravisit::walk_foreign_item(self, item)
    }

    fn visit_anon_const(&mut self, c: &'hir AnonConst) {
        self.body_owners.push(c.def_id);
        intravisit::walk_anon_const(self, c)
    }

    fn visit_inline_const(&mut self, c: &'hir ConstBlock) {
        self.body_owners.push(c.def_id);
        intravisit::walk_inline_const(self, c)
    }

    fn visit_expr(&mut self, ex: &'hir Expr<'hir>) {
        if let ExprKind::Closure(closure) = ex.kind {
            self.body_owners.push(closure.def_id);
        }
        intravisit::walk_expr(self, ex)
    }

    fn visit_trait_item(&mut self, item: &'hir TraitItem<'hir>) {
        if Node::TraitItem(item).associated_body().is_some() {
            self.body_owners.push(item.owner_id.def_id);
        }

        self.trait_items.push(item.trait_item_id());
        intravisit::walk_trait_item(self, item)
    }

    fn visit_impl_item(&mut self, item: &'hir ImplItem<'hir>) {
        if Node::ImplItem(item).associated_body().is_some() {
            self.body_owners.push(item.owner_id.def_id);
        }

        self.impl_items.push(item.impl_item_id());
        intravisit::walk_impl_item(self, item)
    }
}
