//! Runs several rustdoc lints, consolidating them into a single pass for
//! efficiency and simplicity.

mod bare_urls;
mod check_code_block_syntax;
mod html_tags;
mod redundant_explicit_links;
mod unescaped_backticks;

use super::Pass;
use crate::clean::*;
use crate::core::DocContext;
use crate::visit::DocVisitor;

pub(crate) const RUN_LINTS: Pass =
    Pass { name: "run-lints", run: Some(run_lints), description: "runs some of rustdoc's lints" };

struct Linter<'a, 'tcx> {
    cx: &'a mut DocContext<'tcx>,
}

pub(crate) fn run_lints(krate: Crate, cx: &mut DocContext<'_>) -> Crate {
    Linter { cx }.visit_crate(&krate);
    krate
}

impl DocVisitor<'_> for Linter<'_, '_> {
    fn visit_item(&mut self, item: &Item) {
        let Some(hir_id) = DocContext::as_local_hir_id(self.cx.tcx, item.item_id) else {
            // If non-local, no need to check anything.
            return;
        };
        let dox = item.doc_value();
        if !dox.is_empty() {
            let may_have_link = dox.contains(&[':', '['][..]);
            let may_have_block_comment_or_html = dox.contains(['<', '>']);
            // ~~~rust
            // // This is a real, supported commonmark syntax for block code
            // ~~~
            let may_have_code = dox.contains(&['~', '`', '\t'][..]) || dox.contains("    ");
            if may_have_link {
                bare_urls::visit_item(self.cx, item, hir_id, &dox);
                redundant_explicit_links::visit_item(self.cx, item, hir_id);
            }
            if may_have_code {
                check_code_block_syntax::visit_item(self.cx, item, &dox);
                unescaped_backticks::visit_item(self.cx, item, hir_id, &dox);
            }
            if may_have_block_comment_or_html {
                html_tags::visit_item(self.cx, item, hir_id, &dox);
            }
        }

        self.visit_item_recur(item)
    }
}
