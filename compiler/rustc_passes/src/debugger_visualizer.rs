//! Detecting usage of the `#[debugger_visualizer]` attribute.

use hir::CRATE_HIR_ID;
use rustc_data_structures::fx::FxHashSet;
use rustc_expand::base::resolve_path;
use rustc_hir as hir;
use rustc_hir::def_id::CrateNum;
use rustc_hir::itemlikevisit::ItemLikeVisitor;
use rustc_hir::{HirId, Target};
use rustc_middle::ty::query::Providers;
use rustc_middle::ty::TyCtxt;
use rustc_span::def_id::LOCAL_CRATE;
use rustc_span::{sym, DebuggerVisualizerFile, DebuggerVisualizerType};

use std::sync::Arc;

struct DebuggerVisualizerCollector<'tcx> {
    debugger_visualizers: FxHashSet<DebuggerVisualizerFile>,
    tcx: TyCtxt<'tcx>,
}

impl<'v, 'tcx> ItemLikeVisitor<'v> for DebuggerVisualizerCollector<'tcx> {
    fn visit_item(&mut self, item: &hir::Item<'_>) {
        let target = Target::from_item(item);
        match target {
            Target::Mod => {
                self.check_for_debugger_visualizer(item.hir_id());
            }
            _ => {}
        }
    }

    fn visit_trait_item(&mut self, _: &hir::TraitItem<'_>) {}

    fn visit_impl_item(&mut self, _: &hir::ImplItem<'_>) {}

    fn visit_foreign_item(&mut self, _: &hir::ForeignItem<'_>) {}
}

impl<'tcx> DebuggerVisualizerCollector<'tcx> {
    fn new(tcx: TyCtxt<'tcx>) -> DebuggerVisualizerCollector<'tcx> {
        DebuggerVisualizerCollector { tcx, debugger_visualizers: FxHashSet::default() }
    }

    fn check_for_debugger_visualizer(&mut self, hir_id: HirId) {
        let attrs = self.tcx.hir().attrs(hir_id);
        for attr in attrs {
            if attr.has_name(sym::debugger_visualizer) {
                let list = match attr.meta_item_list() {
                    Some(list) => list,
                    _ => continue,
                };

                let meta_item = match list.len() {
                    1 => match list[0].meta_item() {
                        Some(meta_item) => meta_item,
                        _ => continue,
                    },
                    _ => continue,
                };

                let file = match (meta_item.name_or_empty(), meta_item.value_str()) {
                    (sym::natvis_file, Some(value)) => {
                        match resolve_path(&self.tcx.sess.parse_sess, value.as_str(), attr.span) {
                            Ok(file) => file,
                            Err(mut err) => {
                                err.emit();
                                continue;
                            }
                        }
                    }
                    (_, _) => continue,
                };

                if file.is_file() {
                    let contents = match std::fs::read(&file) {
                        Ok(contents) => contents,
                        Err(err) => {
                            self.tcx
                                .sess
                                .struct_span_err(
                                    attr.span,
                                    &format!(
                                        "Unable to read contents of file `{}`. {}",
                                        file.display(),
                                        err
                                    ),
                                )
                                .emit();
                            continue;
                        }
                    };

                    self.debugger_visualizers.insert(DebuggerVisualizerFile::new(
                        Arc::from(contents),
                        DebuggerVisualizerType::Natvis,
                    ));
                } else {
                    self.tcx
                        .sess
                        .struct_span_err(
                            attr.span,
                            &format!("{} is not a valid file", file.display()),
                        )
                        .emit();
                }
            }
        }
    }
}

/// Traverses and collects the debugger visualizers for a specific crate.
fn debugger_visualizers<'tcx>(tcx: TyCtxt<'tcx>, cnum: CrateNum) -> Vec<DebuggerVisualizerFile> {
    assert_eq!(cnum, LOCAL_CRATE);

    // Initialize the collector.
    let mut collector = DebuggerVisualizerCollector::new(tcx);

    // Collect debugger visualizers in this crate.
    tcx.hir().visit_all_item_likes(&mut collector);

    // Collect debugger visualizers on the crate attributes.
    collector.check_for_debugger_visualizer(CRATE_HIR_ID);

    // Extract out the found debugger_visualizer items.
    let DebuggerVisualizerCollector { debugger_visualizers, .. } = collector;

    let mut visualizers = debugger_visualizers.into_iter().collect::<Vec<_>>();

    // Sort the visualizers so we always get a deterministic query result.
    visualizers.sort();
    visualizers
}

pub fn provide(providers: &mut Providers) {
    providers.debugger_visualizers = debugger_visualizers;
}
