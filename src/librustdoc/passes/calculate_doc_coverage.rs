use crate::clean;
use crate::core::DocContext;
use crate::fold::{self, DocFolder};
use crate::passes::Pass;

use syntax::attr;

pub const CALCULATE_DOC_COVERAGE: Pass = Pass {
    name: "calculate-doc-coverage",
    pass: calculate_doc_coverage,
    description: "counts the number of items with and without documentation",
};

fn calculate_doc_coverage(krate: clean::Crate, _: &DocContext<'_, '_, '_>) -> clean::Crate {
    let mut calc = CoverageCalculator::default();
    let krate = calc.fold_crate(krate);

    let total_minus_traits = calc.total - calc.total_trait_impls;
    let docs_minus_traits = calc.with_docs - calc.trait_impls_with_docs;

    print!("Rustdoc found {}/{} items with documentation", calc.with_docs, calc.total);
    println!(" ({}/{} not counting trait impls)", docs_minus_traits, total_minus_traits);

    if calc.total > 0 {
        let percentage = (calc.with_docs as f64 * 100.0) / calc.total as f64;
        let percentage_minus_traits =
            (docs_minus_traits as f64 * 100.0) / total_minus_traits as f64;
        println!("    Score: {:.1}% ({:.1}% not counting trait impls)",
                 percentage, percentage_minus_traits);
    }

    krate
}

#[derive(Default)]
struct CoverageCalculator {
    total: usize,
    with_docs: usize,
    total_trait_impls: usize,
    trait_impls_with_docs: usize,
}

impl fold::DocFolder for CoverageCalculator {
    fn fold_item(&mut self, i: clean::Item) -> Option<clean::Item> {
        match i.inner {
            clean::StrippedItem(..) => {}
            clean::ImplItem(ref impl_)
                if attr::contains_name(&i.attrs.other_attrs, "automatically_derived")
                    || impl_.synthetic || impl_.blanket_impl.is_some() =>
            {
                // skip counting anything inside these impl blocks
                // FIXME(misdreavus): need to also find items that came out of a derive macro
                return Some(i);
            }
            // non-local items are skipped because they can be out of the users control, especially
            // in the case of trait impls, which rustdoc eagerly inlines
            _ => if i.def_id.is_local() {
                let has_docs = !i.attrs.doc_strings.is_empty();

                if let clean::ImplItem(ref i) = i.inner {
                    if let Some(ref tr) = i.trait_ {
                        debug!("counting impl {:#} for {:#}", tr, i.for_);

                        self.total += 1;
                        if has_docs {
                            self.with_docs += 1;
                        }

                        // trait impls inherit their docs from the trait definition, so documenting
                        // them can be considered optional

                        self.total_trait_impls += 1;
                        if has_docs {
                            self.trait_impls_with_docs += 1;
                        }

                        for it in &i.items {
                            self.total_trait_impls += 1;
                            if !it.attrs.doc_strings.is_empty() {
                                self.trait_impls_with_docs += 1;
                            }
                        }
                    } else {
                        // inherent impls *can* be documented, and those docs show up, but in most
                        // cases it doesn't make sense, as all methods on a type are in one single
                        // impl block
                        debug!("not counting impl {:#}", i.for_);
                    }
                } else {
                    debug!("counting {} {:?}", i.type_(), i.name);
                    self.total += 1;
                    if has_docs {
                        self.with_docs += 1;
                    }
                }
            }
        }

        self.fold_item_recur(i)
    }
}
