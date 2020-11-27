use rustc_data_structures::fx::{FxHashMap, FxHashSet};
use rustc_errors::Applicability;
use rustc_hir as hir;
use rustc_hir::def_id::{DefId, LocalDefId, LOCAL_CRATE};
use rustc_hir::itemlikevisit::ItemLikeVisitor;
use rustc_middle::ty::TyCtxt;
use rustc_session::lint;
use rustc_span::{Span, Symbol};

pub fn check_crate(tcx: TyCtxt<'_>) {
    let mut used_trait_imports = FxHashSet::default();
    for &body_id in tcx.hir().krate().bodies.keys() {
        let item_def_id = tcx.hir().body_owner_def_id(body_id);
        let imports = tcx.used_trait_imports(item_def_id);
        debug!("GatherVisitor: item_def_id={:?} with imports {:#?}", item_def_id, imports);
        used_trait_imports.extend(imports.iter());
    }

    let mut visitor = CheckVisitor { tcx, used_trait_imports };
    tcx.hir().krate().visit_all_item_likes(&mut visitor);

    unused_crates_lint(tcx);
}

impl ItemLikeVisitor<'v> for CheckVisitor<'tcx> {
    fn visit_item(&mut self, item: &hir::Item<'_>) {
        if item.vis.node.is_pub() || item.span.is_dummy() {
            return;
        }
        if let hir::ItemKind::Use(ref path, _) = item.kind {
            self.check_import(item.hir_id, path.span);
        }
    }

    fn visit_trait_item(&mut self, _trait_item: &hir::TraitItem<'_>) {}

    fn visit_impl_item(&mut self, _impl_item: &hir::ImplItem<'_>) {}

    fn visit_foreign_item(&mut self, _foreign_item: &hir::ForeignItem<'_>) {}
}

struct CheckVisitor<'tcx> {
    tcx: TyCtxt<'tcx>,
    used_trait_imports: FxHashSet<LocalDefId>,
}

impl CheckVisitor<'tcx> {
    fn check_import(&self, id: hir::HirId, span: Span) {
        let def_id = self.tcx.hir().local_def_id(id);
        if !self.tcx.maybe_unused_trait_import(def_id) {
            return;
        }

        if self.used_trait_imports.contains(&def_id) {
            return;
        }

        self.tcx.struct_span_lint_hir(lint::builtin::UNUSED_IMPORTS, id, span, |lint| {
            let msg = if let Ok(snippet) = self.tcx.sess.source_map().span_to_snippet(span) {
                format!("unused import: `{}`", snippet)
            } else {
                "unused import".to_owned()
            };
            lint.build(&msg).emit();
        });
    }
}

fn unused_crates_lint(tcx: TyCtxt<'_>) {
    let lint = lint::builtin::UNUSED_EXTERN_CRATES;

    // Collect first the crates that are completely unused.  These we
    // can always suggest removing (no matter which edition we are
    // in).
    let unused_extern_crates: FxHashMap<LocalDefId, Span> = tcx
        .maybe_unused_extern_crates(LOCAL_CRATE)
        .iter()
        .filter(|&&(def_id, _)| {
            // The `def_id` here actually was calculated during resolution (at least
            // at the time of this writing) and is being shipped to us via a side
            // channel of the tcx. There may have been extra expansion phases,
            // however, which ended up removing the `def_id` *after* expansion such
            // as the `ReplaceBodyWithLoop` pass (which is a bit of a hack, but hey)
            //
            // As a result we need to verify that `def_id` is indeed still valid for
            // our AST and actually present in the HIR map. If it's not there then
            // there's safely nothing to warn about, and otherwise we carry on with
            // our execution.
            //
            // Note that if we carry through to the `extern_mod_stmt_cnum` query
            // below it'll cause a panic because `def_id` is actually bogus at this
            // point in time otherwise.
            if tcx.hir().find(tcx.hir().local_def_id_to_hir_id(def_id)).is_none() {
                return false;
            }
            true
        })
        .filter(|&&(def_id, _)| {
            tcx.extern_mod_stmt_cnum(def_id).map_or(true, |cnum| {
                !tcx.is_compiler_builtins(cnum)
                    && !tcx.is_panic_runtime(cnum)
                    && !tcx.has_global_allocator(cnum)
                    && !tcx.has_panic_handler(cnum)
            })
        })
        .cloned()
        .collect();

    // Collect all the extern crates (in a reliable order).
    let mut crates_to_lint = vec![];
    tcx.hir().krate().visit_all_item_likes(&mut CollectExternCrateVisitor {
        tcx,
        crates_to_lint: &mut crates_to_lint,
    });

    for extern_crate in &crates_to_lint {
        let def_id = extern_crate.def_id.expect_local();
        let id = tcx.hir().local_def_id_to_hir_id(def_id);
        let item = tcx.hir().expect_item(id);

        // If the crate is fully unused, we suggest removing it altogether.
        // We do this in any edition.
        if extern_crate.warn_if_unused {
            if let Some(&span) = unused_extern_crates.get(&def_id) {
                tcx.struct_span_lint_hir(lint, id, span, |lint| {
                    // Removal suggestion span needs to include attributes (Issue #54400)
                    let span_with_attrs = tcx
                        .get_attrs(extern_crate.def_id)
                        .iter()
                        .map(|attr| attr.span)
                        .fold(span, |acc, attr_span| acc.to(attr_span));

                    lint.build("unused extern crate")
                        .span_suggestion_short(
                            span_with_attrs,
                            "remove it",
                            String::new(),
                            Applicability::MachineApplicable,
                        )
                        .emit();
                });
                continue;
            }
        }

        // If we are not in Rust 2018 edition, then we don't make any further
        // suggestions.
        if !tcx.sess.rust_2018() {
            continue;
        }

        // If the extern crate isn't in the extern prelude,
        // there is no way it can be written as an `use`.
        let orig_name = extern_crate.orig_name.unwrap_or(item.ident.name);
        if !tcx.extern_prelude.get(&orig_name).map_or(false, |from_item| !from_item) {
            continue;
        }

        // If the extern crate is renamed, then we cannot suggest replacing it with a use as this
        // would not insert the new name into the prelude, where other imports in the crate may be
        // expecting it.
        if extern_crate.orig_name.is_some() {
            continue;
        }

        // If the extern crate has any attributes, they may have funky
        // semantics we can't faithfully represent using `use` (most
        // notably `#[macro_use]`). Ignore it.
        if !tcx.get_attrs(extern_crate.def_id).is_empty() {
            continue;
        }
        tcx.struct_span_lint_hir(lint, id, extern_crate.span, |lint| {
            // Otherwise, we can convert it into a `use` of some kind.
            let base_replacement = match extern_crate.orig_name {
                Some(orig_name) => format!("use {} as {};", orig_name, item.ident.name),
                None => format!("use {};", item.ident.name),
            };
            let vis = tcx.sess.source_map().span_to_snippet(item.vis.span).unwrap_or_default();
            let add_vis = |to| if vis.is_empty() { to } else { format!("{} {}", vis, to) };
            lint.build("`extern crate` is not idiomatic in the new edition")
                .span_suggestion_short(
                    extern_crate.span,
                    &format!("convert it to a `{}`", add_vis("use".to_string())),
                    add_vis(base_replacement),
                    Applicability::MachineApplicable,
                )
                .emit();
        })
    }
}

struct CollectExternCrateVisitor<'a, 'tcx> {
    tcx: TyCtxt<'tcx>,
    crates_to_lint: &'a mut Vec<ExternCrateToLint>,
}

struct ExternCrateToLint {
    /// `DefId` of the extern crate
    def_id: DefId,

    /// span from the item
    span: Span,

    /// if `Some`, then this is renamed (`extern crate orig_name as
    /// crate_name`), and -- perhaps surprisingly -- this stores the
    /// *original* name (`item.name` will contain the new name)
    orig_name: Option<Symbol>,

    /// if `false`, the original name started with `_`, so we shouldn't lint
    /// about it going unused (but we should still emit idiom lints).
    warn_if_unused: bool,
}

impl<'a, 'tcx, 'v> ItemLikeVisitor<'v> for CollectExternCrateVisitor<'a, 'tcx> {
    fn visit_item(&mut self, item: &hir::Item<'_>) {
        if let hir::ItemKind::ExternCrate(orig_name) = item.kind {
            let extern_crate_def_id = self.tcx.hir().local_def_id(item.hir_id);
            self.crates_to_lint.push(ExternCrateToLint {
                def_id: extern_crate_def_id.to_def_id(),
                span: item.span,
                orig_name,
                warn_if_unused: !item.ident.as_str().starts_with('_'),
            });
        }
    }

    fn visit_trait_item(&mut self, _trait_item: &hir::TraitItem<'_>) {}

    fn visit_impl_item(&mut self, _impl_item: &hir::ImplItem<'_>) {}

    fn visit_foreign_item(&mut self, _foreign_item: &hir::ForeignItem<'_>) {}
}
