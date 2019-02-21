use crate::clean;
use crate::core::DocContext;
use crate::html::item_type::ItemType;
use crate::fold::{self, DocFolder};
use crate::passes::Pass;

use syntax::attr;

use std::collections::BTreeMap;
use std::fmt;
use std::ops;

pub const CALCULATE_DOC_COVERAGE: Pass = Pass {
    name: "calculate-doc-coverage",
    pass: calculate_doc_coverage,
    description: "counts the number of items with and without documentation",
};

fn calculate_doc_coverage(krate: clean::Crate, _: &DocContext<'_, '_, '_>) -> clean::Crate {
    let mut calc = CoverageCalculator::default();
    let krate = calc.fold_crate(krate);

    calc.print_results();

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

impl ops::Sub for ItemCount {
    type Output = Self;

    fn sub(self, rhs: Self) -> Self {
        ItemCount {
            total: self.total - rhs.total,
            with_docs: self.with_docs - rhs.with_docs,
        }
    }
}

impl ops::AddAssign for ItemCount {
    fn add_assign(&mut self, rhs: Self) {
        self.total += rhs.total;
        self.with_docs += rhs.with_docs;
    }
}

impl fmt::Display for ItemCount {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}/{}", self.with_docs, self.total)
    }
}

#[derive(Default)]
struct CoverageCalculator {
    items: BTreeMap<ItemType, ItemCount>,
}

impl CoverageCalculator {
    fn print_results(&self) {
        use crate::html::item_type::ItemType::*;

        let mut total = ItemCount::default();

        let main_types = [
            Module, Function,
            Struct, StructField,
            Enum, Variant,
            Union,
            Method,
            Trait, TyMethod,
            AssociatedType, AssociatedConst,
            Macro,
            Static, Constant,
            ForeignType, Existential,
            Typedef, TraitAlias,
            Primitive, Keyword,
        ];

        println!("+-{0:->25}-+-{0:->10}-+-{0:->10}-+-{0:->10}-+", "");
        println!("| {:<25} | {:>10} | {:>10} | {:>10} |",
                 "Item Type", "Documented", "Total", "Percentage");
        println!("+-{0:->25}-+-{0:->10}-+-{0:->10}-+-{0:->10}-+", "");

        for item_type in &main_types {
            let count = self.items.get(item_type).cloned().unwrap_or_default();

            if let Some(percentage) = count.percentage() {
                println!("| {:<25} | {:>10} | {:>10} | {:>9.1}% |",
                         table_name(item_type), count.with_docs, count.total, percentage);

                total += count;
            }
        }

        println!("+-{0:->25}-+-{0:->10}-+-{0:->10}-+-{0:->10}-+", "");

        if let Some(count) = self.items.get(&Impl) {
            if let Some(percentage) = count.percentage() {
                if let Some(percentage) = total.percentage() {
                    println!("| {:<25} | {:>10} | {:>10} | {:>9.1}% |",
                             "Total (non trait impls)", total.with_docs, total.total, percentage);
                }

                println!("+-{0:->25}-+-{0:->10}-+-{0:->10}-+-{0:->10}-+", "");

                println!("| {:<25} | {:>10} | {:>10} | {:>9.1}% |",
                         table_name(&Impl), count.with_docs, count.total, percentage);

                println!("+-{0:->25}-+-{0:->10}-+-{0:->10}-+-{0:->10}-+", "");

                total += *count;
            }
        }

        println!("| {:<25} | {:>10} | {:>10} | {:>9.1}% |",
                 "Total", total.with_docs, total.total, total.percentage().unwrap_or(0.0));
        println!("+-{0:->25}-+-{0:->10}-+-{0:->10}-+-{0:->10}-+", "");
    }
}

impl fold::DocFolder for CoverageCalculator {
    fn fold_item(&mut self, mut i: clean::Item) -> Option<clean::Item> {
        let has_docs = !i.attrs.doc_strings.is_empty();

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
                if attr::contains_name(&i.attrs.other_attrs, "automatically_derived")
                    || impl_.synthetic || impl_.blanket_impl.is_some() =>
            {
                // built-in derives get the `#[automatically_derived]` attribute, and
                // synthetic/blanket impls are made up by rustdoc and can't be documented
                // FIXME(misdreavus): need to also find items that came out of a derive macro
                return Some(i);
            }
            clean::ImplItem(ref impl_) => {
                if let Some(ref tr) = impl_.trait_ {
                    debug!("counting impl {:#} for {:#}", tr, impl_.for_);

                    // trait impls inherit their docs from the trait definition, so documenting
                    // them can be considered optional
                    self.items.entry(ItemType::Impl).or_default().count_item(has_docs);

                    for it in &impl_.items {
                        let has_docs = !it.attrs.doc_strings.is_empty();
                        self.items.entry(ItemType::Impl).or_default().count_item(has_docs);
                    }

                    // now skip recursing, so that we don't double-count this impl's items
                    return Some(i);
                } else {
                    // inherent impls *can* be documented, and those docs show up, but in most
                    // cases it doesn't make sense, as all methods on a type are in one single
                    // impl block
                    debug!("not counting impl {:#}", impl_.for_);
                }
            }
            clean::MacroItem(..) | clean::ProcMacroItem(..) => {
                // combine `macro_rules!` macros and proc-macros in the same count
                debug!("counting macro {:?}", i.name);
                self.items.entry(ItemType::Macro).or_default().count_item(has_docs);
            }
            clean::TraitItem(ref mut trait_) => {
                // because both trait methods with a default impl and struct methods are
                // ItemType::Method, we need to properly tag trait methods as TyMethod instead
                debug!("counting trait {:?}", i.name);
                self.items.entry(ItemType::Trait).or_default().count_item(has_docs);

                // since we're not going on to document the crate, it doesn't matter if we discard
                // the item after counting it
                trait_.items.retain(|it| {
                    if it.type_() == ItemType::Method {
                        let has_docs = !it.attrs.doc_strings.is_empty();
                        self.items.entry(ItemType::TyMethod).or_default().count_item(has_docs);
                        false
                    } else {
                        true
                    }
                });
            }
            _ => {
                debug!("counting {} {:?}", i.type_(), i.name);
                self.items.entry(i.type_()).or_default().count_item(has_docs);
            }
        }

        self.fold_item_recur(i)
    }
}

fn table_name(type_: &ItemType) -> &'static str {
        match *type_ {
            ItemType::Module          => "Modules",
            ItemType::Struct          => "Structs",
            ItemType::Union           => "Unions",
            ItemType::Enum            => "Enums",
            ItemType::Function        => "Functions",
            ItemType::Typedef         => "Type Aliases",
            ItemType::Static          => "Statics",
            ItemType::Trait           => "Traits",
            // inherent impls aren't counted, and trait impls get all their items thrown into this
            // counter
            ItemType::Impl            => "Trait Impl Items",
            // even though trait methods with a default impl get cleaned as Method, we convert them
            // to TyMethod when counting
            ItemType::TyMethod        => "Trait Methods",
            ItemType::Method          => "Methods",
            ItemType::StructField     => "Struct Fields",
            ItemType::Variant         => "Enum Variants",
            ItemType::Macro           => "Macros",
            ItemType::Primitive       => "Primitives",
            ItemType::AssociatedType  => "Associated Types",
            ItemType::Constant        => "Constants",
            ItemType::AssociatedConst => "Associated Constants",
            ItemType::ForeignType     => "Foreign Types",
            ItemType::Keyword         => "Keywords",
            ItemType::Existential     => "Existential Types",
            ItemType::TraitAlias      => "Trait Aliases",
            _                         => panic!("unanticipated ItemType: {}", type_),
        }
}
