use crate::clean;
use crate::core::DocContext;
use crate::fold::{self, DocFolder};
use crate::html::markdown::{find_testable_code, ErrorCodes};
use crate::passes::doc_test_lints::{should_have_doc_example, Tests};
use crate::passes::Pass;
use rustc_lint::builtin::MISSING_DOCS;
use rustc_middle::lint::LintSource;
use rustc_session::lint;
use rustc_span::symbol::sym;
use rustc_span::FileName;
use serde::Serialize;

use std::collections::BTreeMap;
use std::ops;

pub const CALCULATE_DOC_COVERAGE: Pass = Pass {
    name: "calculate-doc-coverage",
    run: calculate_doc_coverage,
    description: "counts the number of items with and without documentation",
};

fn calculate_doc_coverage(krate: clean::Crate, ctx: &DocContext<'_>) -> clean::Crate {
    let mut calc = CoverageCalculator::new(ctx);
    let krate = calc.fold_crate(krate);

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
    ctx: &'a DocContext<'b>,
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
    fn new(ctx: &'a DocContext<'b>) -> CoverageCalculator<'a, 'b> {
        CoverageCalculator { items: Default::default(), ctx }
    }

    fn to_json(&self) -> String {
        serde_json::to_string(
            &self
                .items
                .iter()
                .map(|(k, v)| (k.to_string(), v))
                .collect::<BTreeMap<String, &ItemCount>>(),
        )
        .expect("failed to convert JSON data to string")
    }

    fn print_results(&self) {
        let output_format = self.ctx.renderinfo.borrow().output_format;
        if output_format.map(|o| o.is_json()).unwrap_or_else(|| false) {
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
                    &limit_filename_len(file.to_string()),
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

impl<'a, 'b> fold::DocFolder for CoverageCalculator<'a, 'b> {
    fn fold_item(&mut self, i: clean::Item) -> Option<clean::Item> {
        match i.inner {
            _ if !i.def_id.is_local() => {
                // non-local items are skipped because they can be out of the users control,
                // especially in the case of trait impls, which rustdoc eagerly inlines
                return Some(i);
            }
            clean::StrippedItem(..) => {
                // don't count items in stripped modules
                return Some(i);
            }
            clean::ImportItem(..) | clean::ExternCrateItem(..) => {
                // docs on `use` and `extern crate` statements are not displayed, so they're not
                // worth counting
                return Some(i);
            }
            clean::ImplItem(ref impl_)
                if i.attrs
                    .other_attrs
                    .iter()
                    .any(|item| item.has_name(sym::automatically_derived))
                    || impl_.synthetic
                    || impl_.blanket_impl.is_some() =>
            {
                // built-in derives get the `#[automatically_derived]` attribute, and
                // synthetic/blanket impls are made up by rustdoc and can't be documented
                // FIXME(misdreavus): need to also find items that came out of a derive macro
                return Some(i);
            }
            clean::ImplItem(ref impl_) => {
                if let Some(ref tr) = impl_.trait_ {
                    debug!(
                        "impl {:#} for {:#} in {}",
                        tr.print(),
                        impl_.for_.print(),
                        i.source.filename
                    );

                    // don't count trait impls, the missing-docs lint doesn't so we shouldn't
                    // either
                    return Some(i);
                } else {
                    // inherent impls *can* be documented, and those docs show up, but in most
                    // cases it doesn't make sense, as all methods on a type are in one single
                    // impl block
                    debug!("impl {:#} in {}", impl_.for_.print(), i.source.filename);
                }
            }
            _ => {
                let has_docs = !i.attrs.doc_strings.is_empty();
                let mut tests = Tests { found_tests: 0 };

                find_testable_code(
                    &i.attrs
                        .doc_strings
                        .iter()
                        .map(|d| d.doc.as_str())
                        .collect::<Vec<_>>()
                        .join("\n"),
                    &mut tests,
                    ErrorCodes::No,
                    false,
                    None,
                );

                let has_doc_example = tests.found_tests != 0;
                let hir_id = self.ctx.tcx.hir().local_def_id_to_hir_id(i.def_id.expect_local());
                let (level, source) = self.ctx.tcx.lint_level_at_node(MISSING_DOCS, hir_id);
                // `missing_docs` is allow-by-default, so don't treat this as ignoring the item
                // unless the user had an explicit `allow`
                let should_have_docs =
                    level != lint::Level::Allow || matches!(source, LintSource::Default);
                debug!("counting {:?} {:?} in {}", i.type_(), i.name, i.source.filename);
                self.items.entry(i.source.filename.clone()).or_default().count_item(
                    has_docs,
                    has_doc_example,
                    should_have_doc_example(self.ctx, &i),
                    should_have_docs,
                );
            }
        }

        self.fold_item_recur(i)
    }
}
