use crate::lint_without_lint_pass::is_lint_ref_type;
use clippy_utils::diagnostics::span_lint_and_help;
use regex::Regex;
use rustc_hir::{Attribute, Item, ItemKind, Mutability};
use rustc_lint::{LateContext, LateLintPass};
use rustc_lint_defs::declare_tool_lint;
use rustc_session::impl_lint_pass;

declare_tool_lint! {
    /// ### What it does
    /// Checks if lint formulations have a standardized format.
    ///
    /// ### Why is this bad?
    /// It's not necessarily bad, but we try to enforce a standard in Clippy.
    ///
    /// ### Example
    /// `Checks for use...` can be written as `Checks for usage...` .
    pub clippy::ALMOST_STANDARD_LINT_FORMULATION,
    Warn,
    "lint formulations must have a standardized format.",
    report_in_external_macro: true
}

impl_lint_pass!(AlmostStandardFormulation => [ALMOST_STANDARD_LINT_FORMULATION]);

pub struct AlmostStandardFormulation {
    standard_formulations: Vec<StandardFormulations<'static>>,
}

#[derive(Debug)]
struct StandardFormulations<'a> {
    wrong_pattern: Regex,
    correction: &'a str,
}

impl AlmostStandardFormulation {
    pub fn new() -> Self {
        let standard_formulations = vec![StandardFormulations {
            wrong_pattern: Regex::new("^(Check for|Detects? uses?)").unwrap(),
            correction: "Checks for",
        }];
        Self { standard_formulations }
    }
}

impl<'tcx> LateLintPass<'tcx> for AlmostStandardFormulation {
    fn check_item(&mut self, cx: &LateContext<'tcx>, item: &'tcx Item<'_>) {
        let mut check_next = false;
        if let ItemKind::Static(_, ty, Mutability::Not, _) = item.kind {
            let lines = cx
                .tcx
                .hir_attrs(item.hir_id())
                .iter()
                .filter_map(|attr| Attribute::doc_str(attr).map(|sym| (sym, attr)));
            if is_lint_ref_type(cx, ty) {
                for (line, attr) in lines {
                    let cur_line = line.as_str().trim();
                    if check_next && !cur_line.is_empty() {
                        for formulation in &self.standard_formulations {
                            let starts_with_correct_formulation = cur_line.starts_with(formulation.correction);
                            if !starts_with_correct_formulation && formulation.wrong_pattern.is_match(cur_line) {
                                if let Some(ident) = attr.ident() {
                                    span_lint_and_help(
                                        cx,
                                        ALMOST_STANDARD_LINT_FORMULATION,
                                        ident.span,
                                        "non-standard lint formulation",
                                        None,
                                        format!("consider using `{}`", formulation.correction),
                                    );
                                }
                                return;
                            }
                        }
                        return;
                    } else if cur_line.contains("What it does") {
                        check_next = true;
                    } else if cur_line.contains("Why is this bad") {
                        // Formulation documentation is done. Can add check to ensure that missing formulation is added
                        // and add a check if it matches no accepted formulation
                        return;
                    }
                }
            }
        }
    }
}
