//@rustc-env:CLIPPY_CONF_DIR=tests/ui-toml/arbitrary_source_item_ordering/alphabetical_or_trait_item_order
//@check-pass
#![deny(clippy::arbitrary_source_item_ordering)]

struct Languages;
struct Language;
struct Iter<'a, T>(&'a [&'a T]);

impl<'a> Iterator for Iter<'a, Language> {
    type Item = &'a Language;
    fn next(&mut self) -> Option<Self::Item> {
        todo!()
    }
}

// Follows the trait definition order (but not alphabetical order):
// accepted, because either ordering is allowed in this mode.
impl<'a> IntoIterator for &'a Languages {
    type Item = &'a Language;
    type IntoIter = Iter<'a, Language>;
    fn into_iter(self) -> Self::IntoIter {
        todo!()
    }
}

struct AlsoLanguages;

// Follows the alphabetical order (but not the trait definition order):
// also accepted in this mode.
impl<'a> IntoIterator for &'a AlsoLanguages {
    type IntoIter = Iter<'a, Language>;
    type Item = &'a Language;
    fn into_iter(self) -> Self::IntoIter {
        todo!()
    }
}

// Bare impls only have the alphabetical ordering to compare against.
struct BareImpl;
impl BareImpl {
    fn a() {}
    fn b() {}
    fn c() {}
}

fn main() {}
