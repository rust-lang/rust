use crate::clean;
use crate::core::DocContext;
use crate::fold::{self, DocFolder};
use crate::passes::Pass;

use syntax::attr;

use std::ops::Sub;
use std::fmt;

pub const CALCULATE_DOC_COVERAGE: Pass = Pass {
    name: "calculate-doc-coverage",
    pass: calculate_doc_coverage,
    description: "counts the number of items with and without documentation",
};

fn calculate_doc_coverage(krate: clean::Crate, _: &DocContext<'_, '_, '_>) -> clean::Crate {
    let mut calc = CoverageCalculator::default();
    let krate = calc.fold_crate(krate);

    let non_traits = calc.items - calc.trait_impl_items;

    print!("Rustdoc found {} items with documentation", calc.items);
    println!(" ({} not counting trait impls)", non_traits);

    if let (Some(percentage), Some(percentage_non_traits)) =
        (calc.items.percentage(), non_traits.percentage())
    {
        println!("    Score: {:.1}% ({:.1}% not counting trait impls)",
                 percentage, percentage_non_traits);
    }

    krate
}

#[derive(Default, Copy, Clone)]
struct ItemCount {
    total: u64,
    with_docs: u64,
}

impl ItemCount {
    fn count_item(&mut self, has_docs: bool) {
        self.total += 1;

        if has_docs {
            self.with_docs += 1;
        }
    }

    fn percentage(&self) -> Option<f64> {
        if self.total > 0 {
            Some((self.with_docs as f64 * 100.0) / self.total as f64)
        } else {
            None
        }
    }
}

impl Sub for ItemCount {
    type Output = Self;

    fn sub(self, rhs: Self) -> Self {
        ItemCount {
            total: self.total - rhs.total,
            with_docs: self.with_docs - rhs.with_docs,
        }
    }
}

impl fmt::Display for ItemCount {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}/{}", self.with_docs, self.total)
    }
}

#[derive(Default)]
struct CoverageCalculator {
    items: ItemCount,
    trait_impl_items: ItemCount,
}

impl fold::DocFolder for CoverageCalculator {
    fn fold_item(&mut self, i: clean::Item) -> Option<clean::Item> {
        match i.inner {
            clean::StrippedItem(..) => {
                // don't count items in stripped modules
                return Some(i);
            }
            clean::ImportItem(..) | clean::ExternCrateItem(..) => {}
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

                        self.items.count_item(has_docs);

                        // trait impls inherit their docs from the trait definition, so documenting
                        // them can be considered optional

                        self.trait_impl_items.count_item(has_docs);

                        for it in &i.items {
                            self.trait_impl_items.count_item(!it.attrs.doc_strings.is_empty());
                        }
                    } else {
                        // inherent impls *can* be documented, and those docs show up, but in most
                        // cases it doesn't make sense, as all methods on a type are in one single
                        // impl block
                        debug!("not counting impl {:#}", i.for_);
                    }
                } else {
                    debug!("counting {} {:?}", i.type_(), i.name);
                    self.items.count_item(has_docs);
                }
            }
        }

        self.fold_item_recur(i)
    }
}
