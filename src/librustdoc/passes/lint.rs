//! Runs several rustdoc lints, consolidating them into a single pass for
//! efficiency and simplicity.

mod bare_urls;
mod check_code_block_syntax;
mod html_tags;
mod unescaped_backticks;

use super::Pass;
use crate::clean::*;
use crate::core::DocContext;
use crate::visit::DocVisitor;

pub(crate) const RUN_LINTS: Pass =
    Pass { name: "run-lints", run: run_lints, description: "runs some of rustdoc's lints" };

struct Linter<'a, 'tcx> {
    cx: &'a mut DocContext<'tcx>,
}

pub(crate) fn run_lints(krate: Crate, cx: &mut DocContext<'_>) -> Crate {
    Linter { cx }.visit_crate(&krate);
    krate
}

impl<'a, 'tcx> DocVisitor for Linter<'a, 'tcx> {
    fn visit_item(&mut self, item: &Item) {
        bare_urls::visit_item(self.cx, item);
        check_code_block_syntax::visit_item(self.cx, item);
        html_tags::visit_item(self.cx, item);
        unescaped_backticks::visit_item(self.cx, item);

        self.visit_item_recur(item)
    }
}
