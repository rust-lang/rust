use crate::errors::{ExternCrateNotIdiomatic, UnusedExternCrate};
use rustc_data_structures::fx::FxHashMap;
use rustc_data_structures::unord::UnordSet;
use rustc_hir as hir;
use rustc_hir::def::DefKind;
use rustc_hir::def_id::{DefId, LocalDefId};
use rustc_middle::ty::TyCtxt;
use rustc_session::lint;
use rustc_span::{Span, Symbol};

pub fn check_crate(tcx: TyCtxt<'_>) {
    let mut used_trait_imports: UnordSet<LocalDefId> = Default::default();

    for item_def_id in tcx.hir().body_owners() {
        let imports = tcx.used_trait_imports(item_def_id);
        debug!("GatherVisitor: item_def_id={:?} with imports {:#?}", item_def_id, imports);
        used_trait_imports.extend(imports.items().copied());
    }

    for &id in tcx.maybe_unused_trait_imports(()) {
        debug_assert_eq!(tcx.def_kind(id), DefKind::Use);
        if tcx.visibility(id).is_public() {
            continue;
        }
        if used_trait_imports.contains(&id) {
            continue;
        }
        let item = tcx.hir().expect_item(id);
        if item.span.is_dummy() {
            continue;
        }
        let (path, _) = item.expect_use();
        let msg = if let Ok(snippet) = tcx.sess.source_map().span_to_snippet(path.span) {
            format!("unused import: `{}`", snippet)
        } else {
            "unused import".to_owned()
        };
        tcx.struct_span_lint_hir(
            lint::builtin::UNUSED_IMPORTS,
            item.hir_id(),
            path.span,
            msg,
            |lint| lint,
        );
    }

    unused_crates_lint(tcx);
}

fn unused_crates_lint(tcx: TyCtxt<'_>) {
    let lint = lint::builtin::UNUSED_EXTERN_CRATES;

    // Collect first the crates that are completely unused. These we
    // can always suggest removing (no matter which edition we are
    // in).
    let unused_extern_crates: FxHashMap<LocalDefId, Span> = tcx
        .maybe_unused_extern_crates(())
        .iter()
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

    for id in tcx.hir().items() {
        if matches!(tcx.def_kind(id.owner_id), DefKind::ExternCrate) {
            let item = tcx.hir().item(id);
            if let hir::ItemKind::ExternCrate(orig_name) = item.kind {
                crates_to_lint.push(ExternCrateToLint {
                    def_id: item.owner_id.to_def_id(),
                    span: item.span,
                    orig_name,
                    warn_if_unused: !item.ident.as_str().starts_with('_'),
                });
            }
        }
    }

    let extern_prelude = &tcx.resolutions(()).extern_prelude;

    for extern_crate in &crates_to_lint {
        let def_id = extern_crate.def_id.expect_local();
        let item = tcx.hir().expect_item(def_id);

        // If the crate is fully unused, we suggest removing it altogether.
        // We do this in any edition.
        if extern_crate.warn_if_unused {
            if let Some(&span) = unused_extern_crates.get(&def_id) {
                // Removal suggestion span needs to include attributes (Issue #54400)
                let id = tcx.hir().local_def_id_to_hir_id(def_id);
                let span_with_attrs = tcx
                    .hir()
                    .attrs(id)
                    .iter()
                    .map(|attr| attr.span)
                    .fold(span, |acc, attr_span| acc.to(attr_span));

                tcx.emit_spanned_lint(lint, id, span, UnusedExternCrate { span: span_with_attrs });
                continue;
            }
        }

        // If we are not in Rust 2018 edition, then we don't make any further
        // suggestions.
        if !tcx.sess.rust_2018() {
            continue;
        }

        // If the extern crate isn't in the extern prelude,
        // there is no way it can be written as a `use`.
        let orig_name = extern_crate.orig_name.unwrap_or(item.ident.name);
        if !extern_prelude.get(&orig_name).map_or(false, |from_item| !from_item) {
            continue;
        }

        // If the extern crate is renamed, then we cannot suggest replacing it with a use as this
        // would not insert the new name into the prelude, where other imports in the crate may be
        // expecting it.
        if extern_crate.orig_name.is_some() {
            continue;
        }

        let id = tcx.hir().local_def_id_to_hir_id(def_id);
        // If the extern crate has any attributes, they may have funky
        // semantics we can't faithfully represent using `use` (most
        // notably `#[macro_use]`). Ignore it.
        if !tcx.hir().attrs(id).is_empty() {
            continue;
        }

        let base_replacement = match extern_crate.orig_name {
            Some(orig_name) => format!("use {} as {};", orig_name, item.ident.name),
            None => format!("use {};", item.ident.name),
        };
        let vis = tcx.sess.source_map().span_to_snippet(item.vis_span).unwrap_or_default();
        let add_vis = |to| if vis.is_empty() { to } else { format!("{} {}", vis, to) };
        tcx.emit_spanned_lint(
            lint,
            id,
            extern_crate.span,
            ExternCrateNotIdiomatic {
                span: extern_crate.span,
                msg_code: add_vis("use".to_string()),
                suggestion_code: add_vis(base_replacement),
            },
        );
    }
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
