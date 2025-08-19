use rustc_hir::HirId;
use rustc_hir::def_id::LocalDefId;
use rustc_middle::ty::TyCtxt;
use rustc_span::sym;

use super::Pass;
use crate::clean::{Attributes, Crate, Item};
use crate::core::DocContext;
use crate::visit::DocVisitor;

pub(crate) const CHECK_DOC_CFG: Pass = Pass {
    name: "check-doc-cfg",
    run: Some(check_doc_cfg),
    description: "checks `#[doc(cfg(...))]` for stability feature and unexpected cfgs",
};

pub(crate) fn check_doc_cfg(krate: Crate, cx: &mut DocContext<'_>) -> Crate {
    let mut checker = DocCfgChecker { cx };
    checker.visit_crate(&krate);
    krate
}

struct RustdocCfgMatchesLintEmitter<'a>(TyCtxt<'a>, HirId);

impl<'a> rustc_attr_parsing::CfgMatchesLintEmitter for RustdocCfgMatchesLintEmitter<'a> {
    fn emit_span_lint(
        &self,
        sess: &rustc_session::Session,
        lint: &'static rustc_lint::Lint,
        sp: rustc_span::Span,
        builtin_diag: rustc_lint_defs::BuiltinLintDiag,
    ) {
        self.0.node_span_lint(lint, self.1, sp, |diag| {
            rustc_lint::decorate_builtin_lint(sess, Some(self.0), builtin_diag, diag)
        });
    }
}

struct DocCfgChecker<'a, 'tcx> {
    cx: &'a mut DocContext<'tcx>,
}

impl DocCfgChecker<'_, '_> {
    fn check_attrs(&mut self, attrs: &Attributes, did: LocalDefId) {
        let doc_cfgs = attrs
            .other_attrs
            .iter()
            .filter(|attr| attr.has_name(sym::doc))
            .flat_map(|attr| attr.meta_item_list().unwrap_or_default())
            .filter(|attr| attr.has_name(sym::cfg));

        for doc_cfg in doc_cfgs {
            if let Some([cfg_mi]) = doc_cfg.meta_item_list() {
                let _ = rustc_attr_parsing::cfg_matches(
                    cfg_mi,
                    &self.cx.tcx.sess,
                    RustdocCfgMatchesLintEmitter(
                        self.cx.tcx,
                        self.cx.tcx.local_def_id_to_hir_id(did),
                    ),
                    Some(self.cx.tcx.features()),
                );
            }
        }
    }
}

impl DocVisitor<'_> for DocCfgChecker<'_, '_> {
    fn visit_item(&mut self, item: &'_ Item) {
        if let Some(Some(local_did)) = item.def_id().map(|did| did.as_local()) {
            self.check_attrs(&item.attrs, local_did);
        }

        self.visit_item_recur(item);
    }
}
