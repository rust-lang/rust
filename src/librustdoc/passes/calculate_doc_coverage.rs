//! Calculates information used for the --show-coverage flag.
use crate::clean;
use crate::core::DocContext;
use crate::html::markdown::{find_testable_code, ErrorCodes};
use crate::passes::check_doc_test_visibility::{should_have_doc_example, Tests};
use crate::passes::Pass;
use crate::visit::DocVisitor;
use rustc_hir as hir;
use rustc_lint::builtin::MISSING_DOCS;
use rustc_middle::lint::LintLevelSource;
use rustc_session::lint;
use rustc_span::FileName;
use serde::Serialize;

use std::collections::BTreeMap;
use std::ops;

pub(crate) const CALCULATE_DOC_COVERAGE: Pass = Pass {
    name: "calculate-doc-coverage",
    run: calculate_doc_coverage,
    description: "counts the number of items with and without documentation",
};

fn calculate_doc_coverage(krate: clean::Crate, ctx: &mut DocContext<'_>) -> clean::Crate {
    let mut calc = CoverageCalculator { items: Default::default(), ctx };
    calc.visit_crate(&krate);

    calc.print_results();

    krate
}

#[derive(Default, Copy, Clone, Serialize, Debug)]
struct ItemCount {
    total: u64,
    with_docs: u64,
    total_examples: u64,
    with_examples: u64,
}

impl ItemCount {
    fn count_item(
        &mut self,
        has_docs: bool,
        has_doc_example: bool,
        should_have_doc_examples: bool,
        should_have_docs: bool,
    ) {
        if has_docs || should_have_docs {
            self.total += 1;
        }

        if has_docs {
            self.with_docs += 1;
        }
        if should_have_doc_examples || has_doc_example {
            self.total_examples += 1;
        }
        if has_doc_example {
            self.with_examples += 1;
        }
    }

    fn percentage(&self) -> Option<f64> {
        if self.total > 0 {
            Some((self.with_docs as f64 * 100.0) / self.total as f64)
        } else {
            None
        }
    }

    fn examples_percentage(&self) -> Option<f64> {
        if self.total_examples > 0 {
            Some((self.with_examples as f64 * 100.0) / self.total_examples as f64)
        } else {
            None
        }
    }
}

impl ops::Sub for ItemCount {
    type Output = Self;

    fn sub(self, rhs: Self) -> Self {
        ItemCount {
            total: self.total - rhs.total,
            with_docs: self.with_docs - rhs.with_docs,
            total_examples: self.total_examples - rhs.total_examples,
            with_examples: self.with_examples - rhs.with_examples,
        }
    }
}

impl ops::AddAssign for ItemCount {
    fn add_assign(&mut self, rhs: Self) {
        self.total += rhs.total;
        self.with_docs += rhs.with_docs;
        self.total_examples += rhs.total_examples;
        self.with_examples += rhs.with_examples;
    }
}

struct CoverageCalculator<'a, 'b> {
    items: BTreeMap<FileName, ItemCount>,
    ctx: &'a mut DocContext<'b>,
}

fn limit_filename_len(filename: String) -> String {
    let nb_chars = filename.chars().count();
    if nb_chars > 35 {
        "...".to_string()
            + &filename[filename.char_indices().nth(nb_chars - 32).map(|x| x.0).unwrap_or(0)..]
    } else {
        filename
    }
}

impl<'a, 'b> CoverageCalculator<'a, 'b> {
    fn to_json(&self) -> String {
        serde_json::to_string(
            &self
                .items
                .iter()
                .map(|(k, v)| (k.prefer_local().to_string(), v))
                .collect::<BTreeMap<String, &ItemCount>>(),
        )
        .expect("failed to convert JSON data to string")
    }

    fn print_results(&self) {
        let output_format = self.ctx.output_format;
        if output_format.is_json() {
            println!("{}", self.to_json());
            return;
        }
        let mut total = ItemCount::default();

        fn print_table_line() {
            println!("+-{0:->35}-+-{0:->10}-+-{0:->10}-+-{0:->10}-+-{0:->10}-+", "");
        }

        fn print_table_record(
            name: &str,
            count: ItemCount,
            percentage: f64,
            examples_percentage: f64,
        ) {
            println!(
                "| {:<35} | {:>10} | {:>9.1}% | {:>10} | {:>9.1}% |",
                name, count.with_docs, percentage, count.with_examples, examples_percentage,
            );
        }

        print_table_line();
        println!(
            "| {:<35} | {:>10} | {:>10} | {:>10} | {:>10} |",
            "File", "Documented", "Percentage", "Examples", "Percentage",
        );
        print_table_line();

        for (file, &count) in &self.items {
            if let Some(percentage) = count.percentage() {
                print_table_record(
                    &limit_filename_len(file.prefer_local().to_string_lossy().into()),
                    count,
                    percentage,
                    count.examples_percentage().unwrap_or(0.),
                );

                total += count;
            }
        }

        print_table_line();
        print_table_record(
            "Total",
            total,
            total.percentage().unwrap_or(0.0),
            total.examples_percentage().unwrap_or(0.0),
        );
        print_table_line();
    }
}

impl<'a, 'b> DocVisitor for CoverageCalculator<'a, 'b> {
    fn visit_item(&mut self, i: &clean::Item) {
        if !i.item_id.is_local() {
            // non-local items are skipped because they can be out of the users control,
            // especially in the case of trait impls, which rustdoc eagerly inlines
            return;
        }

        match *i.kind {
            clean::StrippedItem(..) => {
                // don't count items in stripped modules
                return;
            }
            // docs on `use` and `extern crate` statements are not displayed, so they're not
            // worth counting
            clean::ImportItem(..) | clean::ExternCrateItem { .. } => {}
            // Don't count trait impls, the missing-docs lint doesn't so we shouldn't either.
            // Inherent impls *can* be documented, and those docs show up, but in most cases it
            // doesn't make sense, as all methods on a type are in one single impl block
            clean::ImplItem(_) => {}
            _ => {
                let has_docs = !i.attrs.doc_strings.is_empty();
                let mut tests = Tests { found_tests: 0 };

                find_testable_code(
                    &i.attrs.collapsed_doc_value().unwrap_or_default(),
                    &mut tests,
                    ErrorCodes::No,
                    false,
                    None,
                );

                let has_doc_example = tests.found_tests != 0;
                let hir_id = DocContext::as_local_hir_id(self.ctx.tcx, i.item_id).unwrap();
                let (level, source) = self.ctx.tcx.lint_level_at_node(MISSING_DOCS, hir_id);

                // In case we have:
                //
                // ```
                // enum Foo { Bar(u32) }
                // // or:
                // struct Bar(u32);
                // ```
                //
                // there is no need to require documentation on the fields of tuple variants and
                // tuple structs.
                let should_be_ignored = i
                    .item_id
                    .as_def_id()
                    .and_then(|def_id| self.ctx.tcx.opt_parent(def_id))
                    .and_then(|def_id| self.ctx.tcx.hir().get_if_local(def_id))
                    .map(|node| {
                        matches!(
                            node,
                            hir::Node::Variant(hir::Variant {
                                data: hir::VariantData::Tuple(_, _, _),
                                ..
                            }) | hir::Node::Item(hir::Item {
                                kind: hir::ItemKind::Struct(hir::VariantData::Tuple(_, _, _), _),
                                ..
                            })
                        )
                    })
                    .unwrap_or(false);

                // `missing_docs` is allow-by-default, so don't treat this as ignoring the item
                // unless the user had an explicit `allow`.
                //
                let should_have_docs = !should_be_ignored
                    && (level != lint::Level::Allow || matches!(source, LintLevelSource::Default));

                if let Some(span) = i.span(self.ctx.tcx) {
                    let filename = span.filename(self.ctx.sess());
                    debug!("counting {:?} {:?} in {:?}", i.type_(), i.name, filename);
                    self.items.entry(filename).or_default().count_item(
                        has_docs,
                        has_doc_example,
                        should_have_doc_example(self.ctx, i),
                        should_have_docs,
                    );
                }
            }
        }

        self.visit_item_recur(i)
    }
}
