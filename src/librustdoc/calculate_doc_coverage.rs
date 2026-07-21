//! Calculates information used for the --show-coverage flag.

use std::collections::BTreeMap;
use std::fs::{File, create_dir_all};
use std::io::{self, BufWriter, Write, stdout};
use std::ops;

use rustc_hir as hir;
use rustc_lint::builtin::MISSING_DOCS;
use rustc_middle::lint::LintLevelSource;
use rustc_span::{FileName, RemapPathScopeComponents};
use serde::Serialize;
use tracing::debug;

use crate::config::{OutputFormat, RenderOptions};
use crate::core::DocContext;
use crate::docfs::PathError;
use crate::error::Error;
use crate::html::markdown::{ErrorCodes, find_testable_code};
use crate::passes::{Tests, should_have_doc_example};
use crate::visit::DocVisitor;
use crate::{clean, try_err};

pub(crate) fn run(
    krate: &clean::Crate,
    ctx: &mut DocContext<'_>,
    options: &RenderOptions,
) -> Result<(), Error> {
    let is_json = ctx.output_format == OutputFormat::CoverageJson;
    let tcx = ctx.tcx;
    let mut calc = CoverageCalculator { items: Default::default(), ctx };
    calc.visit_crate(&krate);

    if options.output_to_stdout {
        calc.print_results(BufWriter::new(stdout().lock()))
            .map_err(|error| Error::new(error, "<stdout>"))
    } else {
        let out_dir = &options.output;
        try_err!(create_dir_all(out_dir), out_dir);
        let name = krate.name(tcx);
        let mut out_file = out_dir.join(name.as_str());
        out_file.set_extension(if is_json { "json" } else { "txt" });
        let buf = try_err!(File::create_buffered(&out_file), out_file);
        calc.print_results(buf).map_err(|error| Error::new(error, &out_file))?;
        println!("Generated output into {out_file:?}");
        Ok(())
    }
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

impl CoverageCalculator<'_, '_> {
    fn to_json(&self) -> String {
        serde_json::to_string(
            &self
                .items
                .iter()
                .map(|(k, v)| (k.display(RemapPathScopeComponents::COVERAGE).to_string(), v))
                .collect::<BTreeMap<String, &ItemCount>>(),
        )
        .expect("failed to convert JSON data to string")
    }

    fn print_results(&self, mut buf: impl Write) -> io::Result<()> {
        let output_format = self.ctx.output_format;
        if output_format == OutputFormat::CoverageJson {
            return writeln!(buf, "{}", self.to_json());
        }
        let mut total = ItemCount::default();

        fn print_table_line(buf: &mut impl Write) -> io::Result<()> {
            writeln!(buf, "+-{0:->35}-+-{0:->10}-+-{0:->10}-+-{0:->10}-+-{0:->10}-+", "")
        }

        fn print_table_record(
            buf: &mut impl Write,
            name: &str,
            count: ItemCount,
            percentage: f64,
            examples_percentage: f64,
        ) -> io::Result<()> {
            writeln!(
                buf,
                "| {name:<35} | {with_docs:>10} | {percentage:>9.1}% | {with_examples:>10} | \
                 {examples_percentage:>9.1}% |",
                with_docs = count.with_docs,
                with_examples = count.with_examples,
            )
        }

        print_table_line(&mut buf)?;
        writeln!(
            buf,
            "| {:<35} | {:>10} | {:>10} | {:>10} | {:>10} |",
            "File", "Documented", "Percentage", "Examples", "Percentage",
        )?;
        print_table_line(&mut buf)?;

        for (file, &count) in &self.items {
            if let Some(percentage) = count.percentage() {
                print_table_record(
                    &mut buf,
                    &limit_filename_len(
                        file.display(RemapPathScopeComponents::COVERAGE).to_string(),
                    ),
                    count,
                    percentage,
                    count.examples_percentage().unwrap_or(0.),
                )?;

                total += count;
            }
        }

        print_table_line(&mut buf)?;
        print_table_record(
            &mut buf,
            "Total",
            total,
            total.percentage().unwrap_or(0.0),
            total.examples_percentage().unwrap_or(0.0),
        )?;
        print_table_line(&mut buf)
    }
}

impl DocVisitor<'_> for CoverageCalculator<'_, '_> {
    fn visit_item(&mut self, i: &clean::Item) {
        if !i.item_id.is_local() {
            // non-local items are skipped because they can be out of the users control,
            // especially in the case of trait impls, which rustdoc eagerly inlines
            return;
        }

        match i.kind {
            clean::StrippedItem(..) => {
                // don't count items in stripped modules
                return;
            }
            clean::PlaceholderImplItem => {
                // The "real" impl items are handled below.
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

                find_testable_code(&i.doc_value(), &mut tests, ErrorCodes::No, None);

                let has_doc_example = tests.found_tests != 0;
                let hir_id = DocContext::as_local_hir_id(self.ctx.tcx, i.item_id).unwrap();
                let level_spec = self.ctx.tcx.lint_level_spec_at_node(MISSING_DOCS, hir_id);

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
                    .and_then(|def_id| self.ctx.tcx.hir_get_if_local(def_id))
                    .map(|node| {
                        matches!(
                            node,
                            hir::Node::Variant(hir::Variant {
                                data: hir::VariantData::Tuple(_, _, _),
                                ..
                            }) | hir::Node::Item(hir::Item {
                                kind: hir::ItemKind::Struct(_, _, hir::VariantData::Tuple(_, _, _)),
                                ..
                            })
                        )
                    })
                    .unwrap_or(false);

                // `missing_docs` is allow-by-default, so don't treat this as ignoring the item
                // unless the user had an explicit `allow`.
                //
                let should_have_docs = !should_be_ignored
                    && (!level_spec.is_allow()
                        || matches!(level_spec.src, LintLevelSource::Default));

                if let Some(span) = i.span(self.ctx.tcx) {
                    let filename = span.filename(self.ctx.sess());
                    debug!("counting {:?} {:?} in {filename:?}", i.type_(), i.name);
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
