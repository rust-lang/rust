use clippy_utils::diagnostics::span_lint_and_help;
use rustc_ast::ast::{Crate, Inline, Item, ItemKind, ModKind};
use rustc_errors::MultiSpan;
use rustc_lint::{EarlyContext, EarlyLintPass, Level, LintContext};
use rustc_middle::lint::LevelAndSource;
use rustc_session::impl_lint_pass;
use rustc_span::{FileName, Span};
use std::collections::BTreeMap;
use std::path::PathBuf;

declare_clippy_lint! {
    /// ### What it does
    /// Checks for files that are included as modules multiple times.
    ///
    /// ### Why is this bad?
    /// Loading a file as a module more than once causes it to be compiled
    /// multiple times, taking longer and putting duplicate content into the
    /// module tree.
    ///
    /// ### Example
    /// ```rust,ignore
    /// // lib.rs
    /// mod a;
    /// mod b;
    /// ```
    /// ```rust,ignore
    /// // a.rs
    /// #[path = "./b.rs"]
    /// mod b;
    /// ```
    ///
    /// Use instead:
    ///
    /// ```rust,ignore
    /// // lib.rs
    /// mod a;
    /// mod b;
    /// ```
    /// ```rust,ignore
    /// // a.rs
    /// use crate::b;
    /// ```
    #[clippy::version = "1.63.0"]
    pub DUPLICATE_MOD,
    suspicious,
    "file loaded as module multiple times"
}

struct Modules {
    local_path: PathBuf,
    spans: Vec<Span>,
    lint_levels: Vec<LevelAndSource>,
}

#[derive(Default)]
pub struct DuplicateMod {
    /// map from the canonicalized path to `Modules`, `BTreeMap` to make the
    /// order deterministic for tests
    modules: BTreeMap<PathBuf, Modules>,
}

impl_lint_pass!(DuplicateMod => [DUPLICATE_MOD]);

impl EarlyLintPass for DuplicateMod {
    fn check_item(&mut self, cx: &EarlyContext<'_>, item: &Item) {
        if let ItemKind::Mod(_, _, ModKind::Loaded(_, Inline::No, mod_spans, _)) = &item.kind
            && let FileName::Real(real) = cx.sess().source_map().span_to_filename(mod_spans.inner_span)
            && let Some(local_path) = real.into_local_path()
            && let Ok(absolute_path) = local_path.canonicalize()
        {
            let modules = self.modules.entry(absolute_path).or_insert(Modules {
                local_path,
                spans: Vec::new(),
                lint_levels: Vec::new(),
            });
            modules.spans.push(item.span_with_attributes());
            modules.lint_levels.push(cx.get_lint_level(DUPLICATE_MOD));
        }
    }

    fn check_crate_post(&mut self, cx: &EarlyContext<'_>, _: &Crate) {
        for Modules {
            local_path,
            spans,
            lint_levels,
        } in self.modules.values()
        {
            if spans.len() < 2 {
                continue;
            }

            // At this point the lint would be emitted
            assert_eq!(spans.len(), lint_levels.len());
            let spans: Vec<_> = spans
                .iter()
                .zip(lint_levels)
                .filter_map(|(span, lvl)| {
                    if let Some(id) = lvl.lint_id {
                        cx.fulfill_expectation(id);
                    }

                    (!matches!(lvl.level, Level::Allow | Level::Expect)).then_some(*span)
                })
                .collect();

            if spans.len() < 2 {
                continue;
            }

            let mut multi_span = MultiSpan::from_spans(spans.clone());
            let (&first, duplicates) = spans.split_first().unwrap();

            multi_span.push_span_label(first, "first loaded here");
            for &duplicate in duplicates {
                multi_span.push_span_label(duplicate, "loaded again here");
            }

            span_lint_and_help(
                cx,
                DUPLICATE_MOD,
                multi_span,
                format!("file is loaded as a module multiple times: `{}`", local_path.display()),
                None,
                "replace all but one `mod` item with `use` items",
            );
        }
    }
}
