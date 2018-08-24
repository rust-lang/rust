// Copyright 2016 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

use lint;
use rustc::ty::TyCtxt;

use errors::Applicability;
use syntax::ast;
use syntax_pos::Span;

use rustc::hir::def_id::{DefId, LOCAL_CRATE};
use rustc::hir::itemlikevisit::ItemLikeVisitor;
use rustc::hir::print::visibility_qualified;
use rustc::hir;
use rustc::util::nodemap::DefIdSet;

use rustc_data_structures::fx::FxHashMap;

pub fn check_crate<'a, 'tcx>(tcx: TyCtxt<'a, 'tcx, 'tcx>) {
    let mut used_trait_imports = DefIdSet();
    for &body_id in tcx.hir.krate().bodies.keys() {
        let item_def_id = tcx.hir.body_owner_def_id(body_id);
        let imports = tcx.used_trait_imports(item_def_id);
        debug!("GatherVisitor: item_def_id={:?} with imports {:#?}", item_def_id, imports);
        used_trait_imports.extend(imports.iter());
    }

    let mut visitor = CheckVisitor { tcx, used_trait_imports };
    tcx.hir.krate().visit_all_item_likes(&mut visitor);

    unused_crates_lint(tcx);
}

impl<'a, 'tcx, 'v> ItemLikeVisitor<'v> for CheckVisitor<'a, 'tcx> {
    fn visit_item(&mut self, item: &hir::Item) {
        if item.vis.node.is_pub() || item.span.is_dummy() {
            return;
        }
        if let hir::ItemKind::Use(ref path, _) = item.node {
            self.check_import(item.id, path.span);
        }
    }

    fn visit_trait_item(&mut self, _trait_item: &hir::TraitItem) {
    }

    fn visit_impl_item(&mut self, _impl_item: &hir::ImplItem) {
    }
}

struct CheckVisitor<'a, 'tcx: 'a> {
    tcx: TyCtxt<'a, 'tcx, 'tcx>,
    used_trait_imports: DefIdSet,
}

impl<'a, 'tcx> CheckVisitor<'a, 'tcx> {
    fn check_import(&self, id: ast::NodeId, span: Span) {
        let def_id = self.tcx.hir.local_def_id(id);
        if !self.tcx.maybe_unused_trait_import(def_id) {
            return;
        }

        let import_def_id = self.tcx.hir.local_def_id(id);
        if self.used_trait_imports.contains(&import_def_id) {
            return;
        }

        let msg = if let Ok(snippet) = self.tcx.sess.source_map().span_to_snippet(span) {
            format!("unused import: `{}`", snippet)
        } else {
            "unused import".to_string()
        };
        self.tcx.lint_node(lint::builtin::UNUSED_IMPORTS, id, span, &msg);
    }
}

fn unused_crates_lint<'tcx>(tcx: TyCtxt<'_, 'tcx, 'tcx>) {
    let lint = lint::builtin::UNUSED_EXTERN_CRATES;

    // Collect first the crates that are completely unused.  These we
    // can always suggest removing (no matter which edition we are
    // in).
    let unused_extern_crates: FxHashMap<DefId, Span> =
        tcx.maybe_unused_extern_crates(LOCAL_CRATE)
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
            if let Some(id) = tcx.hir.as_local_node_id(def_id) {
                if tcx.hir.find(id).is_none() {
                    return false;
                }
            }
            true
        })
        .filter(|&&(def_id, _)| {
            let cnum = tcx.extern_mod_stmt_cnum(def_id).unwrap();
            !tcx.is_compiler_builtins(cnum)
                && !tcx.is_panic_runtime(cnum)
                && !tcx.has_global_allocator(cnum)
        })
        .cloned()
        .collect();

    // Collect all the extern crates (in a reliable order).
    let mut crates_to_lint = vec![];
    tcx.hir.krate().visit_all_item_likes(&mut CollectExternCrateVisitor {
        tcx,
        crates_to_lint: &mut crates_to_lint,
    });

    for extern_crate in &crates_to_lint {
        assert!(extern_crate.def_id.is_local());

        // If the crate is fully unused, we suggest removing it altogether.
        // We do this in any edition.
        if let Some(&span) = unused_extern_crates.get(&extern_crate.def_id) {
            assert_eq!(extern_crate.def_id.krate, LOCAL_CRATE);
            let hir_id = tcx.hir.definitions().def_index_to_hir_id(extern_crate.def_id.index);
            let id = tcx.hir.hir_to_node_id(hir_id);
            let msg = "unused extern crate";
            tcx.struct_span_lint_node(lint, id, span, msg)
                .span_suggestion_short_with_applicability(
                    span,
                    "remove it",
                    String::new(),
                    Applicability::MachineApplicable)
                .emit();
            continue;
        }

        // If we are not in Rust 2018 edition, then we don't make any further
        // suggestions.
        if !tcx.sess.rust_2018() {
            continue;
        }

        // If the extern crate has any attributes, they may have funky
        // semantics we can't faithfully represent using `use` (most
        // notably `#[macro_use]`). Ignore it.
        if !tcx.get_attrs(extern_crate.def_id).is_empty() {
            continue;
        }

        // Otherwise, we can convert it into a `use` of some kind.
        let hir_id = tcx.hir.definitions().def_index_to_hir_id(extern_crate.def_id.index);
        let id = tcx.hir.hir_to_node_id(hir_id);
        let item = tcx.hir.expect_item(id);
        let msg = "`extern crate` is not idiomatic in the new edition";
        let help = format!(
            "convert it to a `{}`",
            visibility_qualified(&item.vis, "use")
        );
        let base_replacement = match extern_crate.orig_name {
            Some(orig_name) => format!("use {} as {};", orig_name, item.name),
            None => format!("use {};", item.name),
        };
        let replacement = visibility_qualified(&item.vis, &base_replacement);
        tcx.struct_span_lint_node(lint, id, extern_crate.span, msg)
            .span_suggestion_short(extern_crate.span, &help, replacement)
            .emit();
    }
}

struct CollectExternCrateVisitor<'a, 'tcx: 'a> {
    tcx: TyCtxt<'a, 'tcx, 'tcx>,
    crates_to_lint: &'a mut Vec<ExternCrateToLint>,
}

struct ExternCrateToLint {
    /// def-id of the extern crate
    def_id: DefId,

    /// span from the item
    span: Span,

    /// if `Some`, then this is renamed (`extern crate orig_name as
    /// crate_name`), and -- perhaps surprisingly -- this stores the
    /// *original* name (`item.name` will contain the new name)
    orig_name: Option<ast::Name>,
}

impl<'a, 'tcx, 'v> ItemLikeVisitor<'v> for CollectExternCrateVisitor<'a, 'tcx> {
    fn visit_item(&mut self, item: &hir::Item) {
        if let hir::ItemKind::ExternCrate(orig_name) = item.node {
            let extern_crate_def_id = self.tcx.hir.local_def_id(item.id);
            self.crates_to_lint.push(
                ExternCrateToLint {
                    def_id: extern_crate_def_id,
                    span: item.span,
                    orig_name,
                }
            );
        }
    }

    fn visit_trait_item(&mut self, _trait_item: &hir::TraitItem) {
    }

    fn visit_impl_item(&mut self, _impl_item: &hir::ImplItem) {
    }
}
