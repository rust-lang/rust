//@revisions: trait_def
//@rustc-env:CLIPPY_CONF_DIR=tests/ui-toml/arbitrary_source_item_ordering/trait_definition_order
#![deny(clippy::arbitrary_source_item_ordering)]

// A trait whose definition order happens to be alphabetical.
trait AlphaOrder {
    fn a();
    fn b();
    fn c();
}

struct WrongInBothOrders;

// `a, c, b` violates both the alphabetical order and the trait
// definition order, so it is linted in every mode.
impl AlphaOrder for WrongInBothOrders {
    fn a() {}
    fn c() {}
    fn b() {}
    //~^ arbitrary_source_item_ordering
}

// Bare impls have no trait definition to compare against, so the
// trait_item_ordering mode falls back to the alphabetical check.
struct BareImpl;
impl BareImpl {
    fn a() {}
    fn c() {}
    fn b() {}
    //~^ arbitrary_source_item_ordering
}

struct Languages;
struct Language;
struct Iter<'a, T>(&'a [&'a T]);

impl<'a> Iterator for Iter<'a, Language> {
    type Item = &'a Language;
    fn next(&mut self) -> Option<Self::Item> {
        todo!()
    }
}

impl<'a> IntoIterator for &'a Languages {
    fn into_iter(self) -> Self::IntoIter {
        todo!()
    }
    type Item = &'a Language;
    //~^ arbitrary_source_item_ordering
    type IntoIter = Iter<'a, Language>;
}

fn main() {}
